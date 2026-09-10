"""End-to-end tests driving the pipeline against a stub endpoint.

No network and no model: a fake client returns canned structured responses, so
these cover the output header, the bypass short-circuit, and resume.
"""

import asyncio
import csv
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import yaml

from prenomen import cli, pipeline
from prenomen.models import (
    AuthorInput,
    Gender,
    LLMAuthorResponse,
    LLMResponse,
    ReasoningCode,
)

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


class _StubCompletions:
    def __init__(self, calls):
        self._calls = calls

    async def parse(self, *, model, messages, response_format, **kwargs):
        # Record the prompt so a test can assert the model was never asked.
        self._calls.append(messages[0]["content"])

        class _Msg:
            parsed = LLMResponse(
                author=LLMAuthorResponse(
                    first_name="stub",
                    last_name="stub",
                    gender=Gender.FEMALE,
                    confidence=0.42,
                    reasoning_code=ReasoningCode.LINGUISTIC_PATTERN,
                )
            )

        class _Choice:
            message = _Msg()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _StubClient:
    """Stands in for AsyncOpenAI, recording every prompt it is sent."""

    def __init__(self):
        self.calls: list[str] = []
        self.chat = type("chat", (), {"completions": _StubCompletions(self.calls)})()


class PipelineEndToEndTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.client = _StubClient()
        self._real_build_client = pipeline.build_client
        pipeline.build_client = lambda model_config: self.client
        self.addCleanup(self._restore)

    def _restore(self):
        pipeline.build_client = self._real_build_client
        self._tmp.cleanup()

    # -- helpers ----------------------------------------------------------

    def _write_yaml(self, name, obj):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            yaml.safe_dump(obj, f)
        return path

    def _configs(
        self,
        dictionaries=None,
        bypass=0.85,
        model_name="stub-model",
        name_columns=None,
        data_path=None,
        extra_columns=None,
        max_concurrent=4,
        output_dir=None,
    ):
        pipeline_cfg = self._write_yaml(
            "pipeline.yaml",
            {
                "output": {"dir": output_dir or os.path.join(self.tmp, "out")},
                "processing": {
                    "max_concurrent": max_concurrent,
                    "continue": False,
                    "dict_bypass_threshold": bypass,
                },
                "dictionaries": dictionaries or {},
            },
        )
        data_cfg = self._write_yaml(
            "data.yaml",
            {
                "data": {
                    "path": data_path or os.path.join(EXAMPLES, "authors.csv"),
                    "separator": ",",
                    "name_columns": name_columns
                    or [{"first_name": "first_name", "last_name": "last_name"}],
                    "id_column": "author_id",
                    "extra_columns": extra_columns or ["author_id", "countries"],
                }
            },
        )
        model_cfg = self._write_yaml(
            "model.yaml",
            {
                "model": {"name": model_name},
                "api": {"base_url": "http://stub/v1", "api_key": "x"},
                "parameters": {"temperature": 0.0},
            },
        )
        return pipeline_cfg, data_cfg, model_cfg

    def _run(self, extra_argv=(), **kwargs):
        pipeline_cfg, data_cfg, model_cfg = self._configs(**kwargs)
        args = cli.build_parser().parse_args(
            [
                "--config",
                pipeline_cfg,
                "--data-config",
                data_cfg,
                "--model-config",
                model_cfg,
                *extra_argv,
            ]
        )
        return asyncio.run(pipeline.run(args))

    @staticmethod
    def _read(path):
        with open(path) as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def _counts_dictionary(self, rows):
        path = os.path.join(self.tmp, "names.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "male", "female"])
            w.writerows(rows)
        return {
            "test_list": {
                "enabled": True,
                "path": path,
                "format": "counts",
                "family": "test_list",
                "weight": 1.0,
                "separator": ",",
                "name_col": "name",
                "male_col": "male",
                "female_col": "female",
            }
        }

    # -- tests ------------------------------------------------------------

    def test_llm_only_run_writes_one_row_per_author(self):
        out = self._run()
        rows = self._read(out)
        self.assertEqual(len(rows), 25)
        self.assertEqual(len(self.client.calls), 25)
        self.assertTrue(all(r["gender"] == "female" for r in rows))
        self.assertTrue(all(r["dict_bypass"] == "False" for r in rows))

    def test_header_has_no_dictionary_columns_when_none_loaded(self):
        out = self._run()
        with open(out) as f:
            header = f.readline().rstrip("\n").split("\t")
        self.assertEqual(
            header,
            [
                "author_id",
                "countries",
                "first_name",
                "last_name",
                "gender",
                "llm_confidence",
                "reasoning_code",
                "dict_bypass",
            ],
        )

    def test_header_gains_a_column_pair_per_enabled_dictionary(self):
        out = self._run(dictionaries=self._counts_dictionary([["ada", 0, 100]]))
        with open(out) as f:
            header = f.readline().rstrip("\n").split("\t")
        self.assertEqual(header[-2:], ["test_list_gender", "test_list_freq"])

    def test_conclusive_dictionary_entry_skips_the_model(self):
        out = self._run(dictionaries=self._counts_dictionary([["ada", 0, 100]]))
        rows = {r["author_id"]: r for r in self._read(out)}
        ada = rows["A001"]
        self.assertEqual(ada["dict_bypass"], "True")
        self.assertEqual(ada["reasoning_code"], "dict_bypass")
        self.assertEqual(ada["gender"], "female")
        self.assertEqual(ada["test_list_freq"], "1.0000")
        # 25 authors, one of them settled without a model call.
        self.assertEqual(len(self.client.calls), 24)
        self.assertTrue(all("Ada" not in c for c in self.client.calls))

    def test_inconclusive_dictionary_entry_still_calls_the_model(self):
        # 0.60 is below the 0.85 gate, so the model arbitrates and the dictionary
        # match is carried into the output alongside the model answer.
        out = self._run(dictionaries=self._counts_dictionary([["alan", 60, 40]]))
        rows = {r["author_id"]: r for r in self._read(out)}
        alan = rows["A002"]
        self.assertEqual(alan["dict_bypass"], "False")
        self.assertEqual(alan["test_list_gender"], "male")
        self.assertEqual(alan["test_list_freq"], "0.6000")
        self.assertEqual(len(self.client.calls), 25)

    def test_dictionary_evidence_reaches_the_prompt(self):
        self._run(dictionaries=self._counts_dictionary([["alan", 60, 40]]))
        alan_prompt = next(c for c in self.client.calls if "Alan" in c)
        self.assertIn("test_list", alan_prompt)
        self.assertIn("0.60", alan_prompt)

    def test_resume_adds_nothing_on_a_second_run(self):
        out = self._run()
        self.assertEqual(len(self._read(out)), 25)

        self.client.calls.clear()
        out2 = self._run(extra_argv=["--continue"])
        self.assertEqual(out, out2)
        self.assertEqual(len(self._read(out2)), 25)
        self.assertEqual(self.client.calls, [])

    def test_name_cache_answers_a_repeated_name_once(self):
        # Two records, one name: the model is asked once and both rows are written.
        data_path = os.path.join(self.tmp, "dupes.csv")
        with open(data_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["author_id", "first_name", "last_name", "countries"])
            w.writerow(["A1", "Ada", "Lovelace", "GB"])
            w.writerow(["A2", "Ada", "Lovelace", " gb;GB "])
        out = self._run(extra_argv=["--data", data_path])
        self.assertEqual(len(self._read(out)), 2)
        self.assertEqual(len(self.client.calls), 1)

    def test_missing_configured_column_fails_before_any_call(self):
        data_path = os.path.join(self.tmp, "nocountry.csv")
        with open(data_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["author_id", "first_name", "last_name"])
            w.writerow(["A1", "Ada", "Lovelace"])
        with self.assertRaises(ValueError) as ctx:
            self._run(extra_argv=["--data", data_path])
        self.assertIn("countries", str(ctx.exception))
        self.assertEqual(self.client.calls, [])

    # -- regressions ------------------------------------------------------

    def test_a_record_that_raises_does_not_hang_the_run(self):
        # Every worker fails on every record. A worker that died would stop draining
        # the queue and the producer would block on it forever, so this must finish.
        def boom(self, *args, **kwargs):
            raise RuntimeError("dictionary exploded")

        real_query = pipeline.MultiDBLookup.query
        pipeline.MultiDBLookup.query = boom
        self.addCleanup(setattr, pipeline.MultiDBLookup, "query", real_query)

        result = {}
        worker = threading.Thread(
            target=lambda: result.setdefault("out", self._run()), daemon=True
        )
        worker.start()
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive(), "the run hung instead of skipping failures")
        self.assertEqual(self._read(result["out"]), [])

    def test_records_with_a_blank_id_are_all_kept(self):
        # A null id must not collapse every unidentified record onto one dedupe key.
        data_path = os.path.join(self.tmp, "noid.csv")
        with open(data_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["author_id", "first_name", "last_name", "countries"])
            w.writerow(["", "Ada", "Lovelace", "GB"])
            w.writerow(["", "Alan", "Turing", "GB"])
            w.writerow(["A3", "Marie", "Curie", "FR"])
        rows = self._read(self._run(extra_argv=["--data", data_path]))
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            sorted(r["first_name"] for r in rows), ["Ada", "Alan", "Marie"]
        )

    def test_two_name_mappings_yield_a_row_per_author(self):
        # Both authors on a row share that row's id, so a key built from the id alone
        # would drop the second. The second run must then still find both done.
        data_path = os.path.join(self.tmp, "pairs.csv")
        with open(data_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["author_id", "fn1", "ln1", "fn2", "ln2", "countries"])
            w.writerow(["P1", "Ada", "Lovelace", "Alan", "Turing", "GB"])
            w.writerow(["P2", "Marie", "Curie", "Lise", "Meitner", "FR"])
        two = [
            {"first_name": "fn1", "last_name": "ln1"},
            {"first_name": "fn2", "last_name": "ln2"},
        ]
        out = self._run(name_columns=two, data_path=data_path)
        rows = self._read(out)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            sorted(r["first_name"] for r in rows), ["Ada", "Alan", "Lise", "Marie"]
        )

        self.client.calls.clear()
        self._run(name_columns=two, data_path=data_path, extra_argv=["--continue"])
        self.assertEqual(len(self._read(out)), 4)
        self.assertEqual(self.client.calls, [])

    def test_changing_the_dictionary_set_refuses_to_append(self):
        # Appending under the old header would write the new dictionary's values into
        # the old dictionary's columns, which cannot be undone once mixed in one file.
        out = self._run(dictionaries=self._counts_dictionary([["ada", 0, 100]]))
        before = self._read(out)
        with self.assertRaises(ValueError) as ctx:
            self._run()
        self.assertIn("test_list_gender", str(ctx.exception))
        self.assertEqual(self._read(out), before)

    def test_model_name_with_a_slash_stays_one_file(self):
        # Self-hosted model ids are namespaced; an unescaped separator would point at
        # a directory that does not exist.
        out = self._run(model_name="vendor/model-x")
        self.assertEqual(os.path.basename(out), "authors_gender_vendor_model-x.tsv")
        self.assertEqual(len(self._read(out)), 25)

    def _authors_file(self, rows):
        path = os.path.join(self.tmp, "records.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["author_id", "first_name", "last_name", "countries"])
            writer.writerows(rows)
        return path

    def test_leading_zero_ids_stay_distinct_and_zero_resumes(self):
        data = self._authors_file([
            [value, "Ada", "Lovelace", "GB"] for value in ("001", "1", "0")
        ])
        out = self._run(data_path=data)
        self.assertEqual({r["author_id"] for r in self._read(out)}, {"001", "1", "0"})
        before = Path(out).read_bytes()
        self.client.calls.clear()
        self._run(data_path=data, extra_argv=["--continue"])
        self.assertEqual(Path(out).read_bytes(), before)
        self.assertEqual(self.client.calls, [])

    def test_resume_reuses_only_matching_country_context(self):
        data = self._authors_file([["A1", "Andrea", "Smith", "GB;FR"]])
        out = self._run(data_path=data)
        self._authors_file([
            ["A1", "Andrea", "Smith", "GB;FR"],
            ["A2", "Andrea", "Smith", " fr;GB;FR "],
            ["A3", "Andrea", "Smith", "IT"],
        ])
        self.client.calls.clear()
        self._run(data_path=data, extra_argv=["--continue"])
        self.assertEqual(len(self._read(out)), 3)
        self.assertEqual(len(self.client.calls), 1)
        self.assertIn("in: IT", self.client.calls[0])

    def test_unexported_country_reaches_prompt_but_disables_resume_cache(self):
        data = self._authors_file([["A1", "Andrea", "Smith", "NA"]])
        out = self._run(data_path=data, extra_columns=["author_id"])
        self.assertIn("in: NA", self.client.calls[0])
        self.assertNotIn("countries", self._read(out)[0])
        self._authors_file([
            ["A1", "Andrea", "Smith", "NA"],
            ["A2", "Andrea", "Smith", "NA"],
        ])
        self.client.calls.clear()
        self._run(data_path=data, extra_columns=["author_id"], extra_argv=["--continue"])
        self.assertEqual(len(self._read(out)), 2)
        self.assertEqual(len(self.client.calls), 1)

    def test_namibia_country_conditions_dictionary_bypass(self):
        data = self._authors_file([["A1", "Andrea", "Smith", "NA"]])
        dictionary = Path(self.tmp) / "wgnd.csv"
        dictionary.write_text("name,code,gender,wgt\nandrea,NA,F,10\nandrea,IT,M,90\n")
        out = self._run(data_path=data, extra_columns=["author_id"], dictionaries={
            "test": {"path": str(dictionary), "format": "wgnd", "country_aware": True},
        })
        row = self._read(out)[0]
        self.assertEqual(row["gender"], "female")
        self.assertEqual(row["dict_bypass"], "True")
        self.assertEqual(self.client.calls, [])

    def test_corrupt_output_stops_before_calls_in_both_append_modes(self):
        out = self._run()
        with open(out, "a") as f:
            f.write("BROKEN\tGB\tAda\tLovelace\tfemale\t")
        before = Path(out).read_bytes()
        for flag in ("--continue", "--no-continue"):
            with self.subTest(flag=flag):
                self.client.calls.clear()
                with self.assertRaisesRegex(ValueError, "line 27"):
                    self._run(extra_argv=[flag])
                self.assertEqual(self.client.calls, [])
                self.assertEqual(Path(out).read_bytes(), before)

    def test_a_worker_count_below_one_is_refused(self):
        # Zero workers start nothing, drain nothing, and still exit 0 with an empty
        # output file: a typo in the config that looks like a successful run.
        for value in (0, -1, "eight", 1.5, None, True):
            with self.subTest(max_concurrent=value):
                with self.assertRaises(ValueError) as ctx:
                    self._run(max_concurrent=value)
                self.assertIn("max_concurrent", str(ctx.exception))
                self.assertEqual(self.client.calls, [])

    def test_one_worker_is_allowed(self):
        self.assertEqual(len(self._read(self._run(max_concurrent=1))), 25)

    def test_a_home_relative_output_dir_is_expanded(self):
        # Unexpanded, this creates a directory literally named "~" in the cwd.
        home = os.path.join(self.tmp, "home")
        with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
            out = self._run(output_dir="~/results")
        self.assertEqual(os.path.dirname(out), os.path.join(home, "results"))
        self.assertTrue(os.path.exists(out))
        self.assertNotIn("~", out)

    def test_an_interrupted_run_leaves_a_file_that_resumes_cleanly(self):
        # The README's central claim: the output file is the only source of truth,
        # so a run killed mid-flight must leave it valid and pick up where it left
        # off. Nothing tested that end to end.
        state = {"answered": 0, "hang": True}
        reached = asyncio.Event()
        original = self.client.chat.completions.parse

        async def parse(**kwargs):
            state["answered"] += 1
            if state["hang"] and state["answered"] > 5:
                reached.set()
                await asyncio.Event().wait()
            return await original(**kwargs)

        self.client.chat.completions.parse = parse
        cfg, data, model = self._configs()
        argv = ["--config", cfg, "--data-config", data, "--model-config", model]

        async def interrupted():
            task = asyncio.create_task(pipeline.run(cli.build_parser().parse_args(argv)))
            await asyncio.wait_for(reached.wait(), 5)
            await asyncio.sleep(0.05)  # let the writer drain what was answered
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(interrupted())
        out = os.path.join(self.tmp, "out", "authors_gender_stub-model.tsv")
        partial = self._read(out)
        self.assertTrue(0 < len(partial) < 25, f"wrote {len(partial)} of 25")
        self.assertTrue(Path(out).read_bytes().endswith(b"\n"))
        # The half-written file is accepted by the validator, not merely tolerated.
        pipeline.load_done(out, "author_id")

        state["hang"] = False
        self.client.calls.clear()
        resumed = self._read(self._run(extra_argv=["--continue"]))
        self.assertEqual(len(resumed), 25)
        self.assertEqual(len({r["author_id"] for r in resumed}), 25)
        self.assertEqual(len(self.client.calls), 25 - len(partial))

    def test_run_cancellation_cleans_up_requests(self):
        async def scenario():
            started = asyncio.Event()
            stopped = asyncio.Event()

            async def parse(**kwargs):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

            self.client.chat.completions.parse = parse
            cfg, data, model = self._configs()
            args = cli.build_parser().parse_args([
                "--config", cfg, "--data-config", data, "--model-config", model,
            ])
            task = asyncio.create_task(pipeline.run(args))
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(stopped.is_set())
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            self.assertEqual(pending, set())

        asyncio.run(scenario())


class PredictionSharingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.fail = False
        self.null = False
        self.cache = {}
        self.in_flight = {}
        self.queue = asyncio.Queue()

        async def parse(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            self.calls.append(prompt)
            self.started.set()
            await self.release.wait()
            if self.fail:
                raise RuntimeError("endpoint unavailable")
            answer = LLMResponse(author=LLMAuthorResponse(
                first_name="Andrea", last_name="Smith",
                gender=Gender.MALE if "in: IT" in prompt else Gender.FEMALE,
                confidence=0.8, reasoning_code=ReasoningCode.CULTURAL_ORIGIN,
            ))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                parsed=None if self.null else answer,
            ))])

        self.client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(parse=parse),
        ))

    async def asyncTearDown(self):
        tasks = list(self.in_flight.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def infer(self, author_id, countries="GB"):
        await pipeline.infer_gender(
            client=self.client, out_queue=self.queue, model_name="stub",
            author=AuthorInput(first_name="Andrea", last_name="Smith", extra={
                "id": author_id, "countries": countries,
            }),
            prompt_template=pipeline.build_prompt_template(), parameters={},
            dict_lookup=pipeline.MultiDBLookup({}), name_cache=self.cache,
            in_flight=self.in_flight,
        )

    async def test_concurrent_equivalent_contexts_share_one_call_and_keep_rows(self):
        tasks = [asyncio.create_task(self.infer(i, countries)) for i, countries in enumerate(
            ["GB;FR", "fr;gb", " GB;FR;GB ", "FR;GB"],
        )]
        await asyncio.wait_for(self.started.wait(), 2)
        self.assertEqual(len(self.calls), 1)
        self.release.set()
        await asyncio.gather(*tasks)
        ids = {self.queue.get_nowait()[2].extra["id"] for _ in range(4)}
        self.assertEqual(ids, {0, 1, 2, 3})
        await self.infer(4, "FR;GB")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.queue.qsize(), 1)
        self.assertEqual(self.in_flight, {})

    async def test_different_countries_keep_distinct_predictions(self):
        self.release.set()
        await asyncio.gather(self.infer(1, "IT"), self.infer(2, "GB"))
        self.assertEqual(len(self.calls), 2)
        genders = {self.queue.get_nowait()[0].author.gender for _ in range(2)}
        self.assertEqual(genders, {Gender.MALE, Gender.FEMALE})

    async def test_failed_shared_request_is_removed_and_can_be_retried(self):
        self.fail = True
        tasks = [asyncio.create_task(self.infer(i)) for i in range(3)]
        await asyncio.wait_for(self.started.wait(), 2)
        self.release.set()
        await asyncio.gather(*tasks)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.queue.qsize(), 0)
        self.assertEqual(self.cache, {})
        self.assertEqual(self.in_flight, {})
        self.fail = False
        await self.infer(3)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.queue.qsize(), 1)

    async def test_null_response_is_not_cached(self):
        self.null = True
        self.release.set()
        await self.infer(1)
        self.assertEqual(self.cache, {})
        self.assertEqual(self.in_flight, {})
        self.null = False
        await self.infer(2)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.queue.qsize(), 1)

    async def test_timeout_is_not_cached(self):
        with patch.object(pipeline, "REQUEST_TIMEOUT", 0.01):
            await self.infer(1)
        self.assertEqual(self.cache, {})
        self.assertEqual(self.in_flight, {})
        self.release.set()
        await self.infer(2)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.queue.qsize(), 1)

    async def test_cancelling_one_waiter_preserves_the_shared_request(self):
        first = asyncio.create_task(self.infer(1))
        second = asyncio.create_task(self.infer(2))
        await asyncio.wait_for(self.started.wait(), 2)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.release.set()
        await second
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.queue.get_nowait()[2].extra["id"], 2)
        self.assertEqual(self.in_flight, {})


if __name__ == "__main__":
    unittest.main()
