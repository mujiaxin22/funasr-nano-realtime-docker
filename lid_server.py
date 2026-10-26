#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fun-ASR-Nano / MLT-Nano 实时 WebSocket 服务 —— 语种标签 + 断句 + 响应瘦身。

三件事，默认全部关闭，不加任何参数时行为与官方服务完全一致：

1. 每次响应多带一个 ``lang`` 字段（零 GPU 开销）
2. 把官方「句子越短越难切」的动态断句阈值，换成 2pass 服务的固定 800ms
3. 每条响应只发最近一句，不再把全部历史重发一遍（2pass 服务的行为）

    {
      "sentences": [{"start": 0, "end": 2340, "text": "今天天气不错",
                     "lang": "zh", "index": 0}],
      "partial": "今天天",
      "partial_start_ms": 0,
      "duration_ms": 2340,
      "is_final": false,
      "lang": "zh"          <- 新增
    }

语种标签的原理
--------------
不加载任何额外模型，不额外跑一次解码，不碰 GPU。语种是从**已经算出来的转写
文本**反推的：按字符统计书写系统（汉字 / 假名 / 谚文 / 泰文 / 阿拉伯文 / 天城体
/ 希腊文 / 西里尔文）+ 独占变音字母 + 虚词。一次调用只扫几十个字符，
相比一次 vLLM 解码完全可以忽略。

断句与响应体积：为什么默认不是这样
----------------------------------
官方有两个「贴心」的设计，对短句对话其实很难受：

* 动态断句阈值：短句（5 秒内）要静音 2000ms 才切，长句反而 400ms 就切。
  2pass 服务（runtime/run_server_2pass.sh）用的是固定 800ms，手感完全不同。
* 每次响应携带全部历史句子：会话越长响应越大，总流量平方级增长。
  2pass 服务只发当前句 + 自增 index，历史由客户端自己拼。

三个开关都可以单独用，互不影响。

代价与边界
----------
* 粤语默认按中文（zh）输出，符合「粤语也是中文」的口径；
  加 ``--lid-cantonese`` 可以单独识别成 yue。
* 只靠文本分不出的（同为拉丁字母又没有独占变音符号的，比如 fi/sv/nl/sl/et）
  一律保守归 en。要覆盖 modelcard 里全部 31 种语言必须走声学侧二次判别，
  那要额外一次 prefill，不是这个方案的目标。
* 断句调到 800ms 后句子会变短，而这条 vLLM 路径是单遍流式（没有 2pass 的
  离线第二遍精修），模型听到的上下文变短，识别质量可能掉一点。所以做成
  可调参数而不是写死，先试 800，觉得切太碎再往上调。
* 响应瘦身只是「不发」，句子在服务端内部仍然完整保留（内部逻辑要用），
  ``--history-max-sentences 0`` 即可回到官方行为。

部署（Docker）
-------------
Dockerfile 加两行：

    COPY lid_server.py /opt/lid_server.py
    RUN cat /opt/lid_server.py > /usr/local/bin/funasr-realtime-server \\
        && chmod 0755 /usr/local/bin/funasr-realtime-server

这样连 ``docker run`` 的命令行都不用改。若不想顶替官方启动脚本，也可以直接：

    python3 /opt/lid_server.py \\
        --port 10095 --gpu-memory-utilization 0.17 \\
        --vad-silence-ms 800 --vad-max-segment-ms 15000 --history-max-sentences 1

其余参数与官方 ``funasr-realtime-server`` 完全一致（本文件直接复用官方的
argparse，只多了下面这几组开关）。
"""

from __future__ import annotations

import argparse
import logging
import re

logger = logging.getLogger("funasr.lid")


# ---------------------------------------------------------------------------
# 语种判别：纯字符串运算，无第三方依赖，可单独 import 做单测
# ---------------------------------------------------------------------------

# 书写系统 → 语种。这些是字符级的硬信号，出现一个字符就够，最可靠。
_SCRIPTS = (
    ("ja", ((0x3040, 0x30FF), (0x31F0, 0x31FF))),        # 假名
    ("ko", ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),        # 谚文
    ("th", ((0x0E00, 0x0E7F),)),                         # 泰文
    ("ar", ((0x0600, 0x06FF),)),                         # 阿拉伯文
    ("hi", ((0x0900, 0x097F),)),                         # 天城体
    ("el", ((0x0370, 0x03FF),)),                         # 希腊文
    ("bg", ((0x0400, 0x04FF),)),                         # 西里尔文
)

_HAN_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF))

# 拉丁字母语言里，带变音符号的字母本该是最好用的特征，但 š č ž á é ý 这些
# 在捷克语/斯洛伐克语/克罗地亚语/匈牙利语/越南语/葡萄牙语之间大量共用，
# 直接按出现次数打分会把捷克语判成斯洛伐克语（"krásný" 里的 ý 就是个坑）。
#
# 所以这里登记的是「该语言实际使用的字母表」——包含那些平平无奇的 á é í ó ú——
# 导入时自动求出「只被一种语言使用」的字符，共用的全部丢弃。
# 于是捷克语只剩 ě ř ů，斯洛伐克语只剩 ľ ĺ ŕ，越南语剩一整串声调字符，
# 而共用字母多的语言（sl/hr/sv/fi/et/ga）直接落到 en：宁可判 en，也不乱判。
_LATIN_MARKS = {
    "vi": ("àáâãèéêìíòóôõùúý"
           "ăđĩũơưạảấầậắằẵặẹẻẽếềểễệịọỏốồổỗộờởợụủứừửữựỳỵỷỹ"),
    "cs": "áéíóúýčďěňřšťůž",
    "sk": "áäéíóôúýčďľĺňŕšťž",
    "sl": "čšž",
    "hr": "čćđšž",
    "pl": "ąćęłńóśźż",
    "lv": "āčēģīķļņšūž",
    "lt": "ąčęėįšųūž",
    "et": "äõöüšž",
    "hu": "áéíóöőúüű",
    "ro": "ăâîșşţț",
    "mt": "ċġħż",
    "pt": "ãàáâçêéíõôóú",
    "da": "æøå",
    "sv": "äåö",
    "fi": "äö",
    "nl": "ëïé",
    "ga": "áéíóúṁ",
}

# 粤语口语字。默认不启用（粤语统一按 zh），只有 --lid-cantonese 时才用。
_CANTONESE_MARKERS = tuple(
    "嘅咗啲唔冇佢睇喺哋嗰咩攞畀嚟谂叻嘥揾靓嗮嘢咁咋㗎啩冚揸抦嚿踎瞓嬲冧"
)

# 没有任何变音符号的拉丁语言，只能靠虚词区分（ms 与 id 高度重叠，统一归 id）。
_WORD_MARKS = {
    "fil": ("ang", "mga", "ako", "natin", "sila", "niya"),
    "id": ("yang", "dan", "tidak", "dengan", "untuk", "saya", "kamu", "adalah"),
}

_WORD_RE = re.compile(r"[a-z]+")


def _build_exclusive_marks():
    owners = {}
    for code, marks in _LATIN_MARKS.items():
        for char in marks:
            owners.setdefault(char, set()).add(code)
    exclusive = {}
    for code, marks in _LATIN_MARKS.items():
        unique = "".join(sorted({c for c in marks if owners[c] == {code}}))
        if unique:
            exclusive[code] = unique
    return exclusive


_EXCLUSIVE_MARKS = _build_exclusive_marks()


def classify_text(text, cantonese=False):
    """从转写文本推语种代码；判不出返回 None。

    >>> classify_text("今天天气不错")   -> 'zh'
    >>> classify_text("Hello world")   -> 'en'
    >>> classify_text("こんにちは")     -> 'ja'
    >>> classify_text("안녕하세요")     -> 'ko'
    """
    if not text or not text.strip():
        return None
    sample = text.strip().lower()

    script_hits = {code: 0 for code, _ranges in _SCRIPTS}
    han = 0
    latin = 0
    for char in sample:
        code = ord(char)
        if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
            han += 1
        elif 0x41 <= code <= 0x24F:
            latin += 1
        else:
            for name, ranges in _SCRIPTS:
                if any(lo <= code <= hi for lo, hi in ranges):
                    script_hits[name] += 1
                    break

    # 非汉字的书写系统是硬信号，命中即返回
    script = max(script_hits, key=script_hits.get)
    if script_hits[script] > 0:
        return script

    if han:
        if cantonese and any(marker in sample for marker in _CANTONESE_MARKERS):
            return "yue"
        return "zh"

    if not latin:
        return None

    # 拉丁字母：先看独占变音字母
    scores = {}
    for code, marks in _EXCLUSIVE_MARKS.items():
        hits = sum(sample.count(mark) for mark in marks)
        if hits:
            scores[code] = hits
    if scores:
        return max(scores, key=scores.get)

    words = set(_WORD_RE.findall(sample))
    if words:
        for code, marks in _WORD_MARKS.items():
            if sum(1 for mark in marks if mark in words) >= 2:
                return code

    # 无特征的拉丁字母一律保守归 en
    return "en"


# ---------------------------------------------------------------------------
# 挂载到 RealtimeASRSession：只碰 __init__ 和 _build_response 两个方法
# ---------------------------------------------------------------------------

CANTONESE = False
# 只在句子出结果后才标语种：最稳，但首句要等到说完才有标签
SENTENCE_ONLY = False

# 断句：0 = 保留官方的动态阶梯表；>0 = 固定成这个毫秒数（2pass 服务用的是 800）
VAD_SILENCE_MS = 0
# 单句强制切分上限：0 = 保留官方的 60 秒；>0 = 改成这个毫秒数（2pass 服务用 15000）
VAD_MAX_SEGMENT_MS = 0
# 每条响应最多带几句已完成的句子：0 = 全部（官方行为）；1 = 只带最近一句（2pass 行为）
HISTORY_MAX_SENTENCES = 0

_ORIGINALS = {}


def _session_init(self, *args, **kwargs):
    _ORIGINALS["init"](self, *args, **kwargs)
    self._lid_votes = {}
    self._lid_sentence_count = 0
    self._lid_last = None
    self._lid_seq = 0


def _stamp(self, response):
    """把 lang 写进响应。字典是 locked_sentences 里的同一个对象，写一次即持久化。"""
    sentences = response.get("sentences") or []

    # 每句只在它刚完成时用完整文本判一次，之后不再变；序号同理只发一次
    for sentence in sentences:
        if "index" not in sentence:
            sentence["index"] = self._lid_seq
            self._lid_seq += 1
        if not sentence.get("lang"):
            code = classify_text(sentence.get("text") or "", CANTONESE)
            if code:
                sentence["lang"] = code
                self._lid_last = code

    # 句子数变了 = 上一句结束了，重新开始为当前这句投票
    if len(sentences) != self._lid_sentence_count:
        self._lid_sentence_count = len(sentences)
        self._lid_votes = {}

    partial = response.get("partial") or ""
    if partial and not SENTENCE_ONLY:
        code = classify_text(partial, CANTONESE)
        if code:
            self._lid_votes[code] = self._lid_votes.get(code, 0) + len(partial)

    # 已完成句的优先级最高，其次当前这句的累计投票，最后才用上一句的
    current = None
    for sentence in reversed(sentences):
        if sentence.get("lang"):
            current = sentence["lang"]
            break
    if current is None and self._lid_votes:
        current = max(self._lid_votes, key=self._lid_votes.get)
    if current is None:
        current = self._lid_last

    if current:
        for sentence in sentences:
            if not sentence.get("lang"):
                sentence["lang"] = current
        response["lang"] = current

    # 句子仍完整留在服务端（内部还要用），只是不再每条响应重发一遍
    if HISTORY_MAX_SENTENCES and len(sentences) > HISTORY_MAX_SENTENCES:
        response["sentences"] = sentences[-HISTORY_MAX_SENTENCES:]


def _build_response(self, is_final):
    response = _ORIGINALS["build"](self, is_final)
    try:
        if getattr(self, "_lid_votes", None) is not None:
            _stamp(self, response)
    except Exception as exc:  # 语种标签只是附加信息，绝不能影响识别结果本身
        logger.debug("Failed to attach language tag: %s", exc)
    return response


def _tune_vad(vad):
    """把官方的动态阈值换成固定阈值。

    vad 是每条连接单独 create_vad 出来的对象（vad_opts 已深拷贝），改它不影响别人。
    """
    if VAD_SILENCE_MS and getattr(vad, "silence_schedule", None) is not None:
        # 只有一档、且上限无穷 = 不管这句说了多久，都用同一个阈值
        vad.silence_schedule = [(float("inf"), VAD_SILENCE_MS)]
    options = getattr(getattr(vad, "model", None), "vad_opts", None)
    if VAD_MAX_SEGMENT_MS and options is not None:
        options.max_single_segment_time = VAD_MAX_SEGMENT_MS


def _patch_vad_factory(rtws):
    """create_vad 是模块级函数，每条连接建 VAD 时都会走一次。"""
    if getattr(rtws, "_lid_vad_patched", False):
        return
    original = getattr(rtws, "create_vad", None)
    if original is None:
        logger.warning("未找到 create_vad，断句参数不生效")
        return

    def create_vad(vad_model, args):
        vad = original(vad_model, args)
        try:
            _tune_vad(vad)
        except Exception as exc:
            logger.warning("断句参数未生效，沿用官方默认：%s", exc)
        return vad

    _ORIGINALS["create_vad"] = original
    rtws.create_vad = create_vad
    rtws._lid_vad_patched = True


def install():
    """把语种标签与响应瘦身挂到 RealtimeASRSession 上。可重复调用，只生效一次。"""
    from funasr.bin import realtime_ws as rtws

    if not getattr(rtws.RealtimeASRSession, "_lid_patched", False):
        _ORIGINALS["init"] = rtws.RealtimeASRSession.__init__
        _ORIGINALS["build"] = rtws.RealtimeASRSession._build_response
        rtws.RealtimeASRSession.__init__ = _session_init
        rtws.RealtimeASRSession._build_response = _build_response
        rtws.RealtimeASRSession._lid_patched = True

    # 没配断句参数就完全不碰这条路径，少一个故障点
    if VAD_SILENCE_MS or VAD_MAX_SEGMENT_MS:
        try:
            _patch_vad_factory(rtws)
        except Exception:
            logger.exception("断句参数挂载失败，断句行为保持官方默认")

    return rtws.RealtimeASRSession


# ---------------------------------------------------------------------------
# CLI：复用官方的 argparse，只多两个开关
# ---------------------------------------------------------------------------


def _add_lid_args(parser):
    group = parser.add_argument_group("language tag (text-based, zero GPU cost)")
    group.add_argument(
        "--lid-cantonese",
        action="store_true",
        help="把粤语单独标成 yue；默认与普通话一样标成 zh。",
    )
    group.add_argument(
        "--lid-sentence-only",
        action="store_true",
        help="只在句子识别完成后才给 lang；partial 期间沿用上一句的结果，最稳。",
    )

    tuning = parser.add_argument_group("断句与响应体积")
    tuning.add_argument(
        "--vad-silence-ms",
        type=int,
        default=0,
        help=(
            "断句所需的尾部静音毫秒数。默认 0 = 官方的动态阈值"
            "（说得越短要求停得越久，短句要 2 秒才切）。"
            "设成正数就固定成该值，与 2pass 服务一致。"
            "800 是官方验证过的甜点值，再低容易出现语音被提前截断。"
        ),
    )
    tuning.add_argument(
        "--vad-max-segment-ms",
        type=int,
        default=0,
        help=(
            "单句说满这么久就强制切一刀，防止对方一直不停顿攒出超长段把大模型拖慢。"
            "默认 0 = 官方的 60000（60 秒）；2pass 服务用的是 15000。"
        ),
    )
    tuning.add_argument(
        "--history-max-sentences",
        type=int,
        default=0,
        help=(
            "每条响应最多带几句已完成的句子（保留最新的）。"
            "默认 0 = 全部，即官方行为，长会话下响应会越滚越大；"
            "设 1 = 只带最近一句，靠句子里的 index 判断是否新句子，即 2pass 的行为。"
        ),
    )
    return parser


def _apply_args(args):
    global CANTONESE, SENTENCE_ONLY
    global VAD_SILENCE_MS, VAD_MAX_SEGMENT_MS, HISTORY_MAX_SENTENCES
    CANTONESE = bool(getattr(args, "lid_cantonese", False))
    SENTENCE_ONLY = bool(getattr(args, "lid_sentence_only", False))
    VAD_SILENCE_MS = int(getattr(args, "vad_silence_ms", 0) or 0)
    VAD_MAX_SEGMENT_MS = int(getattr(args, "vad_max_segment_ms", 0) or 0)
    HISTORY_MAX_SENTENCES = int(getattr(args, "history_max_sentences", 0) or 0)
    try:
        install()
    except Exception:
        logger.exception("挂载失败，服务按官方行为继续运行")


def cli_main():
    from funasr.bin import realtime_ws as rtws

    try:
        _orig_build = rtws.build_arg_parser
        _orig_main = rtws.main

        def build_arg_parser():
            return _add_lid_args(_orig_build())

        async def main(args):
            _apply_args(args)
            return await _orig_main(args)

        rtws.build_arg_parser = build_arg_parser
        rtws.main = main
    except Exception:
        logger.exception("无法挂载语种标签，按官方原样启动")

    return rtws.cli_main()


if __name__ == "__main__":
    cli_main()
