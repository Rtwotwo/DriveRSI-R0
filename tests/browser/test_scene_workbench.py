"""Optional live-service browser regressions; all browser execution stays on the server."""
import importlib.util
import os
import unittest

URL = os.environ.get("DRIVERSI_WORKBENCH_URL")
AVAILABLE = importlib.util.find_spec("playwright") is not None


@unittest.skipUnless(URL and AVAILABLE, "Requires server browser tooling and explicit workbench URL")
class WorkbenchBrowserTests(unittest.TestCase):
    def setUp(self):
        from playwright.sync_api import sync_playwright
        self.runtime = sync_playwright().start()
        self.addCleanup(self.runtime.stop)
        self.browser = self.runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        self.addCleanup(self.browser.close)
        self.page = self.browser.new_page(viewport={"width": 1550, "height": 1000})
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.goto(URL, wait_until="networkidle")
        self.page.wait_for_function("()=>document.querySelector('#time').textContent.includes(' s ')")

    def tearDown(self):
        self.assertEqual(self.errors, [])

    def test_catalog_and_reference_boundary(self):
        self.assertGreater(int(self.page.locator("#sceneCount").inner_text()), 0)
        self.assertIn("源轨迹参考回放", self.page.locator("#actorDetail").inner_text())
        self.assertIn("未绑定", self.page.locator(".right").inner_text())
        colors = self.page.locator("#map").evaluate("""c => {
            const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
            const s=new Set();for(let i=0;i<d.length;i+=16)s.add(d[i]+','+d[i+1]+','+d[i+2]);
            return s.size;
        }""")
        self.assertGreater(colors, 10)

    def test_timeline_and_scene_switch(self):
        self.page.locator("#seek").evaluate(
            "e=>{e.value=e.max;e.dispatchEvent(new Event('input',{bubbles:true}))}")
        maximum = self.page.locator("#seek").get_attribute("max")
        self.assertEqual(self.page.locator("#seek").input_value(), maximum)
        self.page.locator("#prev").click()
        self.assertEqual(int(self.page.locator("#seek").input_value()), int(maximum) - 1)
        target = self.page.locator(".scene").nth(1).get_attribute("data-id")
        self.page.locator(".scene").nth(1).click()
        self.page.wait_for_function(
            "(x)=>document.querySelector('#sceneTitle').textContent===x", arg=target)
        self.assertEqual(self.page.locator("#seek").input_value(), "0")

    def test_city_filter_and_actor_selection(self):
        values = self.page.locator("#city option").evaluate_all(
            "xs=>xs.map(x=>x.value).filter(Boolean)")
        self.page.locator("#city").select_option(values[-1])
        target = self.page.locator(".scene").first.get_attribute("data-id")
        self.page.locator(".scene").first.click()
        self.page.wait_for_function(
            "(x)=>document.querySelector('#sceneTitle').textContent===x", arg=target)
        if self.page.locator("#actor option").count() > 1:
            self.page.locator("#actor").select_option("1")
            actor = self.page.locator("#actor option:checked").inner_text().split(" · ")[0]
            self.assertIn(actor, self.page.locator("#actorDetail").inner_text())
        self.assertIn("源轨迹参考回放", self.page.locator("#actorDetail").inner_text())
        self.page.locator("#search").fill("NO_SUCH_SCENE_8888")
        self.assertEqual(self.page.locator(".scene").count(), 0)

    def test_playback_uses_elapsed_time(self):
        self.page.locator("#play").click()
        self.page.wait_for_function("()=>Number(document.querySelector('#seek').value)>=5")
        self.page.locator("#play").click()
        value = int(self.page.locator("#seek").input_value())
        self.assertGreaterEqual(value, 5)
        self.page.wait_for_timeout(300)
        self.assertEqual(int(self.page.locator("#seek").input_value()), value)
        self.page.locator("#next").click()
        self.assertEqual(int(self.page.locator("#seek").input_value()), value + 1)

    def test_small_viewport_and_no_horizontal_overflow(self):
        self.page.set_viewport_size({"width": 700, "height": 900})
        self.page.wait_for_timeout(100)
        self.assertTrue(self.page.evaluate(
            "document.documentElement.scrollWidth<=window.innerWidth"))

    def test_loading_scene_does_not_inherit_old_playback_clock(self):
        self.page.locator("#seek").evaluate(
            "e=>{e.value=20;e.dispatchEvent(new Event('input',{bubbles:true}))}")
        pending = []
        self.page.route("**/api/scene?*", lambda route: pending.append(route))
        target = self.page.locator(".scene").nth(1).get_attribute("data-id")
        self.page.locator(".scene").nth(1).click()
        self.page.wait_for_timeout(100)
        self.assertEqual(len(pending), 1)
        self.page.locator("#play").click()
        pending[0].continue_()
        self.page.wait_for_function(
            "(x)=>document.querySelector('#sceneTitle').textContent===x", arg=target)
        self.assertEqual(self.page.locator("#seek").input_value(), "0")
        self.assertEqual(self.page.locator("#play").inner_text(), "播放")

    def test_development_evidence_renders_resource_costs(self):
        import json
        sid = self.page.locator("#sceneTitle").inner_text()
        cost = {"gpu_seconds": 1, "cpu_seconds": 2.5, "llm_tokens": 30,
                "sim_agent_steps": 10, "rendered_frames": 0, "trials": 1}
        fixture = {"experiments": [{
            "plan": {"id": "TEST_ONLY_UI_FIXTURE", "target": "improver",
                     "scene_ids": [sid], "hypothesis": "Synthetic browser fixture, no result claim",
                     "arms": {"baseline": {"version": "r0"}, "candidate": {"version": "r1"}}},
            "attempts": [{"status": "failed"}],
            "cost_by_arm": {"baseline": cost, "candidate": cost},
            "over_budget": {"baseline": [], "candidate": ["llm_tokens"]}}]}
        self.page.route("**/api/experiments", lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(fixture)))
        self.page.reload(wait_until="networkidle")
        self.page.wait_for_function(
            "()=>document.querySelector('#experiments').textContent.includes('TEST_ONLY_UI_FIXTURE')")
        text = self.page.locator("#experiments").inner_text()
        self.assertIn("CPU 2.50 s", text)
        self.assertIn("LLM 30 tokens", text)
        self.assertIn("预算超出", text)
        self.assertIn("未独立裁决", text)

    def test_executor_results_and_polling(self):
        import copy
        import json
        ledger = self.page.request.get(URL + "/api/experiments").json()
        ready = [(entry, job) for entry in ledger["experiments"] for job in entry.get("jobs", [])
                 if job["status"] == "completed" and job.get("evaluation")]
        if not ready:
            self.skipTest("No real executor result registered in this service")
        entry, job = ready[0]
        payload = copy.deepcopy(ledger)
        target = next(j for e in payload["experiments"] for j in e.get("jobs", []) if j["id"] == job["id"])
        target["status"], target["phase"], target["evaluation"] = "running", "training", None
        calls = []
        def reply(route):
            calls.append(1)
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(payload if len(calls) == 1 else ledger))
        self.page.route("**/api/experiments", reply)
        self.page.reload(wait_until="networkidle")
        self.page.locator("#allExperiments").check()
        card = self.page.locator('[data-job-id="' + job["id"] + '"]')
        self.assertEqual(card.get_attribute("data-status"), "running")
        self.page.wait_for_function(
            "(id)=>document.querySelector('[data-job-id=\"'+id+'\"]')?.dataset.status==='completed'",
            arg=job["id"])
        self.assertIn("独立进程评测", card.inner_text())
        self.assertIn("不是官方 PDMS", card.inner_text())
        self.assertIn("GPU 预留", card.inner_text())

    def test_failed_budget_job_retains_cost_and_reason(self):
        import copy
        import json
        ledger = self.page.request.get(URL + "/api/experiments").json()
        ready = [j for e in ledger["experiments"] for j in e.get("jobs", [])]
        if not ready:
            self.skipTest("No executor schema available in this service")
        payload = copy.deepcopy(ledger)
        job = next(j for e in payload["experiments"] for j in e.get("jobs", []))
        job.update(status="budget_exhausted", phase="budget_exhausted", evaluation=None,
                   failure_reason="CPU_BOUNDARY_TEST_ONLY")
        self.page.route("**/api/experiments", lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(payload)))
        self.page.reload(wait_until="networkidle")
        self.page.locator("#allExperiments").check()
        card = self.page.locator('[data-job-id="' + job["id"] + '"]')
        self.assertEqual(card.get_attribute("data-status"), "budget_exhausted")
        self.assertIn("CPU_BOUNDARY_TEST_ONLY", card.inner_text())
        self.assertIn("CPU", card.inner_text())
        self.assertNotIn("独立进程评测", card.inner_text())

    def test_real_campaign_rounds_show_fixed_gate_and_retained_model(self):
        self.page.wait_for_function("()=>document.querySelector('[data-campaign-id]')")
        card = self.page.locator('[data-campaign-id]').first
        text = card.inner_text()
        self.assertIn("hardcase_curriculum_v1", text)
        self.assertIn("固定门控", text)
        self.assertIn("CPU 秒", text)
        self.assertIn("研究模型", text)
        self.assertGreater(card.locator('details[data-round]').count(), 0)
        self.assertIn("最低收益", text)
        self.assertIn("碰撞不退化", text)
        self.assertIn("训练课程", self.page.locator('#campaigns').locator('..').inner_text())
        ledger = self.page.request.get(URL + '/api/experiments').json()
        self.assertTrue(ledger.get('campaigns'))
        for campaign in ledger['campaigns']:
            self.assertNotIn('promotion_slots', campaign)
            self.assertNotIn('spec', campaign)
            self.assertFalse(campaign['approved_teacher'])
        card.locator('details').first.locator('summary').click()
        self.assertTrue(card.locator('details').first.evaluate('e=>e.open'))
        self.page.wait_for_timeout(2200)
        self.assertTrue(card.locator('details').first.evaluate('e=>e.open'))
        self.page.set_viewport_size({'width': 700, 'height': 1000})
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'))

    def test_campaign_promoted_fixture_is_explicit_research_only(self):
        import copy
        import json
        payload = copy.deepcopy(self.page.request.get(URL + '/api/experiments').json())
        if not payload.get('campaigns'):
            self.skipTest('No campaign schema available')
        c = payload['campaigns'][0]
        c['id'] = 'TEST_ONLY_PROMOTION_UI'
        c['incumbent'] = 'TEST_ONLY_CANDIDATE'
        c['history'] = c['history'][:1]
        c['history'][0]['decision']['outcome'] = 'promoted'
        c['history'][0]['decision']['checks'] = {k: True for k in c['history'][0]['decision']['checks']}
        self.page.route('**/api/experiments', lambda route: route.fulfill(status=200,
            content_type='application/json', body=json.dumps(payload)))
        self.page.reload(wait_until='networkidle')
        card = self.page.locator('[data-campaign-id="TEST_ONLY_PROMOTION_UI"]')
        self.assertIn('晋级研究模型', card.inner_text())
        self.assertIn('TEST_ONLY_CANDIDATE', card.inner_text())
        self.assertIn('未声称统计显著性', card.inner_text())

    def test_catalog_poll_removes_newly_protected_visible_scene(self):
        import json
        old = self.page.locator('#sceneTitle').inner_text()
        catalog = self.page.request.get(URL + '/api/catalog').json()
        catalog['scenes'] = [r for r in catalog['scenes'] if r['id'] != old]
        self.page.route('**/api/catalog', lambda route: route.fulfill(status=200,
            content_type='application/json', body=json.dumps(catalog)))
        self.page.wait_for_function('(id)=>document.querySelector("#sceneTitle").textContent!==id', arg=old)
        self.assertEqual(self.page.locator('.scene[data-id="'+old+'"]').count(),0)
        self.assertEqual(self.page.locator('#seek').input_value(),'0')

    def test_quality_snapshot_and_missing_report_are_not_qualification(self):
        import json
        payload = {"status": "evidence_snapshot", "generated_at": "TEST_ONLY_QUALITY_UI",
                   "industrial_ready": False, "components": [
                       {"name": "TEST_ONLY_RENDER", "status": "limited", "evidence": "fixture evidence",
                        "metrics": {"camera_frames": 12, "high_quality": 0},
                        "limits": ["新视角尚未验证", "不是实时性能证明"]}]}
        self.page.route("**/api/quality", lambda route: route.fulfill(status=200,
                        content_type="application/json", body=json.dumps(payload)))
        self.page.reload(wait_until="networkidle")
        panel = self.page.locator("#platform-quality")
        self.assertIn("TEST_ONLY_QUALITY_UI", panel.inner_text())
        self.assertIn("尚未整体达到工业级验收", panel.inner_text())
        self.assertIn("新视角尚未验证", panel.inner_text())
        self.assertIn("有限支持", panel.inner_text())
        self.page.unroute("**/api/quality")
        self.page.route("**/api/quality", lambda route: route.fulfill(status=400,
                        content_type="application/json", body='{"error":"unavailable"}'))
        self.page.reload(wait_until="networkidle")
        self.assertIn("不能视为通过", panel.inner_text())
        self.assertEqual(self.page.locator("[data-quality-component]").count(), 0)
