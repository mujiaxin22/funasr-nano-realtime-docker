#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lid_server.py 的自检：语种判别 + 会话挂载，不需要 GPU / funasr / torch。

    python3 test_lid_server.py
"""

import asyncio
import importlib.util
import pathlib
import sys
import types
import unittest

_HERE = pathlib.Path(__file__).resolve().parent


def _load_module():
    path = _HERE / "lid_server.py"
    spec = importlib.util.spec_from_file_location("lid_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LID = _load_module()


# ---------------------------------------------------------------------------
# 一个复刻官方 _build_response 的假会话
# ---------------------------------------------------------------------------


class _Vad:
    def __init__(self):
        self.current_speech_start = None


class FakeSession:
    """字段与行为对齐 funasr/bin/realtime_ws.py 的 RealtimeASRSession。"""

    def __init__(self):
        self.locked_sentences = []
        self.last_partial_text = ""
        self.last_partial_start_ms = 0
        self.total_samples = 16000
        self.sample_rate = 16000
        self.vad = _Vad()

    def _build_response(self, is_final):
        duration_ms = int(self.total_samples * 1000 / self.sample_rate)
        sentences = list(self.locked_sentences)
        partial = self.last_partial_text
        if partial:
            partial_start = self.last_partial_start_ms
        elif self.vad.current_speech_start is not None:
            partial_start = self.vad.current_speech_start
        else:
            partial_start = duration_ms
        if is_final:
            return {
                "sentences": sentences, "partial": "", "partial_start_ms": 0,
                "duration_ms": duration_ms, "is_final": True,
            }
        return {
            "sentences": sentences, "partial": partial,
            "partial_start_ms": partial_start,
            "duration_ms": duration_ms, "is_final": False,
        }


def _install_fake_funasr():
    """让 lid_server.install() 能在没有 funasr 的环境里跑起来。"""
    package = types.ModuleType("funasr")
    package.__path__ = []
    sub = types.ModuleType("funasr.bin")
    sub.__path__ = []
    realtime = types.ModuleType("funasr.bin.realtime_ws")
    realtime.RealtimeASRSession = FakeSession
    sys.modules.setdefault("funasr", package)
    sys.modules["funasr.bin"] = sub
    sys.modules["funasr.bin.realtime_ws"] = realtime
    package.bin = sub
    sub.realtime_ws = realtime
    return realtime


def _new_session():
    _install_fake_funasr()
    LID.install()
    return FakeSession()


def _complete(session, text, start=0, end=2000):
    session.locked_sentences.append({"text": text, "start": start, "end": end})


class _FakeVadOpts:
    def __init__(self):
        self.max_single_segment_time = 60000


class _FakeVad:
    """模拟 DynamicStreamingVAD 里我们关心的那几个字段。"""

    def __init__(self):
        self.silence_schedule = [(5000, 2000), (10000, 1500), (float("inf"), 100)]
        self.model = types.SimpleNamespace(vad_opts=_FakeVadOpts())


class _PlainVad:
    """--endpoint-mode client 时返回的对象，压根没有这些字段。"""
    pass


class _BrokenVad:
    """字段访问就抛错的对象，用来验证打不上参数时不会连累连接建立。"""

    @property
    def silence_schedule(self):
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------


class TestClassifyText(unittest.TestCase):
    def test_cjk_is_chinese(self):
        self.assertEqual(LID.classify_text("今天天气不错"), "zh")
        self.assertEqual(LID.classify_text("我们一起去吃饭吧"), "zh")

    def test_cantonese_is_chinese_by_default(self):
        # 粤语口语字默认不单独识别，按中文算
        self.assertEqual(LID.classify_text("我哋今日去边度食饭"), "zh")
        self.assertEqual(LID.classify_text("佢哋已经走咗啦"), "zh")

    def test_cantonese_optional(self):
        self.assertEqual(LID.classify_text("我哋今日去边度食饭", True), "yue")
        self.assertEqual(LID.classify_text("佢哋已经走咗啦", True), "yue")
        # 书面化的粤语本来就该是中文
        self.assertEqual(LID.classify_text("他们今天去哪里吃饭", True), "zh")

    def test_english(self):
        self.assertEqual(LID.classify_text("Hello world"), "en")
        self.assertEqual(LID.classify_text("Good morning everyone"), "en")

    def test_scripts(self):
        self.assertEqual(LID.classify_text("こんにちは"), "ja")
        self.assertEqual(LID.classify_text("안녕하세요"), "ko")
        self.assertEqual(LID.classify_text("สวัสดีครับ"), "th")
        self.assertEqual(LID.classify_text("مرحبا بالعالم"), "ar")
        self.assertEqual(LID.classify_text("नमस्ते दुनिया"), "hi")
        self.assertEqual(LID.classify_text("Καλημέρα κόσμε"), "el")
        self.assertEqual(LID.classify_text("Привет мир"), "bg")

    def test_latin_with_exclusive_diacritics(self):
        self.assertEqual(LID.classify_text("Hôm nay trời đẹp quá"), "vi")
        self.assertEqual(LID.classify_text("Dobře, to je velmi hezké"), "cs")
        self.assertEqual(LID.classify_text("Ďakujem, to je veľmi pekné"), "sk")
        self.assertEqual(LID.classify_text("Dzisiaj jest ładny dzień"), "pl")
        self.assertEqual(LID.classify_text("A bűvös erdő"), "hu")
        self.assertEqual(LID.classify_text("Aș dori să cumpăr"), "ro")
        self.assertEqual(LID.classify_text("Il-ġurnata llum"), "mt")
        self.assertEqual(LID.classify_text("O açúcar está bom"), "pt")
        self.assertEqual(LID.classify_text("Rød grød med fløde"), "da")

    def test_latin_without_exclusive_diacritics_is_english(self):
        # 捷克语/斯洛伐克语共用 š/č/ž/ý，芬兰语/瑞典语共用 ä/ö，
        # 克罗地亚语/斯洛文尼亚语共用 č/ć/š/ž：文本侧拒绝乱猜，一律保守归 en
        self.assertEqual(LID.classify_text("Dnes je krasny den"), "en")
        self.assertEqual(LID.classify_text("Dnes je krásný den"), "en")
        self.assertEqual(LID.classify_text("Vädret är bra idag"), "en")
        self.assertEqual(LID.classify_text("Dobar dan, kako ste"), "en")
        # 匈牙利语没有 ő/ű 时同样认不出来
        self.assertEqual(LID.classify_text("Jo reggelt kivanok"), "en")

    def test_word_marks(self):
        self.assertEqual(LID.classify_text("Saya pergi ke pasar dengan kamu"), "id")
        self.assertEqual(LID.classify_text("Magandang umaga sa inyong lahat"), "en")

    def test_empty(self):
        self.assertIsNone(LID.classify_text(""))
        self.assertIsNone(LID.classify_text("   "))
        self.assertIsNone(LID.classify_text(None))
        self.assertIsNone(LID.classify_text("。，！？"))

    def test_mixed_language_prefers_speech(self):
        # 中文夹英文词，按中文算
        self.assertEqual(LID.classify_text("我买了一个 iPhone 手机"), "zh")

    def test_uppercase_and_spaces(self):
        self.assertEqual(LID.classify_text("  HELLO  WORLD  "), "en")

    def test_exclusive_marks_are_really_exclusive(self):
        for code, marks in LID._EXCLUSIVE_MARKS.items():
            for char in marks:
                owners = [
                    other
                    for other, other_marks in LID._LATIN_MARKS.items()
                    if char in other_marks
                ]
                self.assertEqual(owners, [code], f"{char!r} 同时属于 {owners}")


class TestSessionStamping(unittest.TestCase):
    def setUp(self):
        self.session = _new_session()

    def _partial(self, text):
        self.session.last_partial_text = text
        return self.session._build_response(False)

    def test_partial_carries_lang(self):
        response = self._partial("今天天气")
        self.assertEqual(response["lang"], "zh")
        self.assertEqual(response["partial"], "今天天气")
        self.assertEqual(response["is_final"], False)

    def test_original_fields_unchanged(self):
        response = self._partial("今天天气")
        for key in ("sentences", "partial", "partial_start_ms", "duration_ms", "is_final"):
            self.assertIn(key, response)

    def test_sentence_lang_is_stamped_once_and_persisted(self):
        _complete(self.session, "今天天气不错")
        response = self._partial("今天天气")
        self.assertEqual(response["sentences"][0]["lang"], "zh")
        self.assertEqual(response["lang"], "zh")
        # 响应里的 dict 就是 locked_sentences 里的同一个对象，写一次即持久化
        self.assertEqual(self.session.locked_sentences[0]["lang"], "zh")

    def test_english_sentence(self):
        _complete(self.session, "Hello world")
        response = self._partial("Hello wo")
        self.assertEqual(response["lang"], "en")
        self.assertEqual(response["sentences"][0]["lang"], "en")

    def test_final_response_uses_last_sentence(self):
        _complete(self.session, "今天天气不错")
        response = self.session._build_response(True)
        self.assertEqual(response["is_final"], True)
        self.assertEqual(response["partial"], "")
        self.assertEqual(response["lang"], "zh")
        self.assertEqual(response["sentences"][0]["lang"], "zh")

    def test_votes_resist_leading_english_word(self):
        # 首句以英文词开头：partial 先显示 en，字数积累后稳定到 zh
        response = self._partial("iPhone")
        self.assertEqual(response["lang"], "en")
        response = self._partial("iPhone 今天发")
        self.assertEqual(response["lang"], "zh")
        response = self._partial("iPhone 今天发布了新机型")
        self.assertEqual(response["lang"], "zh")

    def test_votes_reset_on_new_sentence(self):
        self._partial("iPhone")
        self.assertEqual(self.session._lid_votes, {"en": 6})
        _complete(self.session, "iPhone 今天发布了新机型")
        # 下一次构建响应时才发现句子数变了，随即清空上一句的投票
        self._partial("")
        self.assertEqual(self.session._lid_votes, {})

    def test_switch_follows_completed_sentences(self):
        # 语种切换要等新句子说完才生效，避免句子中途标签来回抖
        _complete(self.session, "今天天气不错")
        self.assertEqual(self._partial("Good morning")["lang"], "zh")
        _complete(self.session, "Good morning")
        self.assertEqual(self._partial("Good morn")["lang"], "en")

    def test_last_lang_survives_sentence_clear(self):
        # COMMIT 模式会清空 locked_sentences，标签不应随之消失
        _complete(self.session, "今天天气不错")
        self._partial("今天天气")
        self.session.locked_sentences = []
        response = self._partial("今天")
        self.assertEqual(response["lang"], "zh")

    def test_no_signal_means_no_lang_field(self):
        response = self._partial("。")
        self.assertNotIn("lang", response)
        self.assertEqual(response["sentences"], [])

    def test_speaker_split_inherits_lang(self):
        # 说话人重聚会把句子按时间切开，切出来的新句子要继承语种
        _complete(self.session, "今天天气不错", 0, 2000)
        self._partial("今天天气")
        original = self.session.locked_sentences[0]
        self.session.locked_sentences = [
            {"text": "今天天气", "start": 0, "end": 1200},
            {"text": "不错", "start": 1200, "end": 2000},
        ]
        response = self._partial("")
        self.assertEqual(response["lang"], "zh")
        for sentence in response["sentences"]:
            self.assertEqual(sentence["lang"], "zh")
        self.assertEqual(original["lang"], "zh")


class TestSentenceOnlyMode(unittest.TestCase):
    def setUp(self):
        LID.SENTENCE_ONLY = True
        self.session = _new_session()

    def tearDown(self):
        LID.SENTENCE_ONLY = False

    def test_no_lang_before_first_sentence_completes(self):
        self.session.last_partial_text = "今天天气"
        response = self.session._build_response(False)
        self.assertNotIn("lang", response)

    def test_lang_from_completed_sentence(self):
        _complete(self.session, "Hello world")
        response = self.session._build_response(False)
        self.assertEqual(response["lang"], "en")


class TestCantoneseFlag(unittest.TestCase):
    def setUp(self):
        LID.CANTONESE = True
        self.session = _new_session()

    def tearDown(self):
        LID.CANTONESE = False

    def test_cantonese_sentence(self):
        _complete(self.session, "我哋今日去边度食饭")
        response = self.session._build_response(False)
        self.assertEqual(response["lang"], "yue")
        self.assertEqual(response["sentences"][0]["lang"], "yue")


class TestTuning(unittest.TestCase):
    def tearDown(self):
        LID.CANTONESE = False
        LID.SENTENCE_ONLY = False
        LID.VAD_SILENCE_MS = 0
        LID.VAD_MAX_SEGMENT_MS = 0
        LID.HISTORY_MAX_SENTENCES = 0

    def test_fixed_silence_replaces_whole_schedule(self):
        vad = _FakeVad()
        LID.VAD_SILENCE_MS = 800
        LID._tune_vad(vad)
        # 只有一档且上限无穷 = 不管这句说了多久都是 800ms
        self.assertEqual(vad.silence_schedule, [(float("inf"), 800)])

    def test_max_segment_is_applied(self):
        vad = _FakeVad()
        LID.VAD_MAX_SEGMENT_MS = 15000
        LID._tune_vad(vad)
        self.assertEqual(vad.model.vad_opts.max_single_segment_time, 15000)

    def test_nothing_touched_when_disabled(self):
        vad = _FakeVad()
        LID._tune_vad(vad)
        self.assertEqual(len(vad.silence_schedule), 3)
        self.assertEqual(vad.model.vad_opts.max_single_segment_time, 60000)

    def test_vad_without_those_fields_is_ignored(self):
        LID.VAD_SILENCE_MS = 800
        LID.VAD_MAX_SEGMENT_MS = 15000
        LID._tune_vad(_PlainVad())

    def test_create_vad_is_wrapped(self):
        realtime = _install_fake_funasr()
        realtime.create_vad = lambda vad_model, args: _FakeVad()
        LID.VAD_SILENCE_MS = 800
        LID.VAD_MAX_SEGMENT_MS = 15000
        LID._patch_vad_factory(realtime)
        vad = realtime.create_vad(None, None)
        self.assertEqual(vad.silence_schedule, [(float("inf"), 800)])
        self.assertEqual(vad.model.vad_opts.max_single_segment_time, 15000)

    def test_create_vad_hands_back_original_on_failure(self):
        realtime = _install_fake_funasr()
        realtime.create_vad = lambda vad_model, args: _BrokenVad()
        LID.VAD_SILENCE_MS = 800
        LID._patch_vad_factory(realtime)
        with self.assertLogs(LID.logger, level="WARNING"):
            vad = realtime.create_vad(None, None)
        self.assertIsInstance(vad, _BrokenVad)

    def test_create_vad_untouched_without_tuning(self):
        realtime = _install_fake_funasr()
        original = lambda vad_model, args: _FakeVad()
        realtime.create_vad = original
        LID.install()
        self.assertIs(getattr(realtime, "create_vad"), original)

    def test_sentence_index_increments(self):
        session = _new_session()
        _complete(session, "今天天气不错")
        session._build_response(False)
        _complete(session, "Hello world")
        response = session._build_response(False)
        self.assertEqual(response["sentences"][0]["index"], 0)
        self.assertEqual(response["sentences"][1]["index"], 1)

    def test_history_trim_keeps_only_latest(self):
        session = _new_session()
        LID.HISTORY_MAX_SENTENCES = 1
        _complete(session, "今天天气不错")
        _complete(session, "Hello world")
        response = session._build_response(False)
        self.assertEqual(len(response["sentences"]), 1)
        self.assertEqual(response["sentences"][0]["text"], "Hello world")
        self.assertEqual(response["sentences"][0]["index"], 1)

    def test_trim_only_shrinks_the_wire_payload(self):
        # 服务端内部仍然完整保留，内部逻辑还要用
        session = _new_session()
        LID.HISTORY_MAX_SENTENCES = 1
        _complete(session, "今天天气不错")
        _complete(session, "Hello world")
        session._build_response(False)
        self.assertEqual(len(session.locked_sentences), 2)
        for sentence in session.locked_sentences:
            self.assertIn("index", sentence)
            self.assertIn("lang", sentence)

    def test_all_history_by_default(self):
        session = _new_session()
        _complete(session, "今天天气不错")
        _complete(session, "Hello world")
        self.assertEqual(len(session._build_response(False)["sentences"]), 2)

    def test_trim_keeps_language_tag(self):
        session = _new_session()
        LID.HISTORY_MAX_SENTENCES = 1
        _complete(session, "今天天气不错")
        _complete(session, "Hello world")
        response = session._build_response(False)
        self.assertEqual(response["sentences"][0]["lang"], "en")
        self.assertEqual(response["lang"], "en")


class TestDegradation(unittest.TestCase):
    def tearDown(self):
        LID.CANTONESE = False
        LID.SENTENCE_ONLY = False

    def test_response_survives_stamp_failure(self):
        session = _new_session()
        # 放进一个非 dict，让打标签逻辑炸掉
        session.locked_sentences.append("boom")
        response = session._build_response(False)
        self.assertEqual(response["sentences"], ["boom"])
        self.assertEqual(response["is_final"], False)
        self.assertNotIn("lang", response)

    def test_missing_hook_falls_back_to_vanilla_cli(self):
        realtime = _install_fake_funasr()
        calls = []

        def vanilla_cli_main():
            calls.append(True)

        # 官方模块里没有 build_arg_parser / main（版本升级改了名字的情况）
        realtime.cli_main = vanilla_cli_main
        realtime.build_arg_parser = None
        realtime.main = None
        del realtime.build_arg_parser
        del realtime.main
        # 报错要被吞掉并记进日志，服务照常起来
        with self.assertLogs(LID.logger, level="ERROR"):
            LID.cli_main()
        self.assertEqual(calls, [True])

    def test_argparse_integration(self):
        # 复刻官方 cli_main：build_arg_parser / main 都是模块级全局名，
        # 所以 lid_server 的替换能被它看到
        realtime = _install_fake_funasr()
        seen = {}

        def build_arg_parser():
            import argparse

            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int, default=10095)
            return parser

        async def main(args):
            seen["args"] = args
            return None

        def cli_main():
            args = realtime.build_arg_parser().parse_args(
                ["--port", "12345", "--lid-cantonese", "--lid-sentence-only"]
            )
            asyncio.run(realtime.main(args))
            return args

        realtime.build_arg_parser = build_arg_parser
        realtime.main = main
        realtime.cli_main = cli_main

        args = LID.cli_main()
        self.assertEqual(args.port, 12345)
        self.assertTrue(args.lid_cantonese)
        self.assertTrue(args.lid_sentence_only)
        self.assertIs(seen["args"], args)

    def test_install_is_idempotent(self):
        _install_fake_funasr()
        LID.install()
        first = FakeSession.__init__
        LID.install()
        self.assertIs(FakeSession.__init__, first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
