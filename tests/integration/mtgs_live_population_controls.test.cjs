/* Explicit live-session acceptance. This resets and advances only the supplied loopback session. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.DRIVERSI_PLAYWRIGHT_MODULE||'playwright');

test('live population identity survives reset, step, intervention and pause', {timeout:120000}, async()=>{
 const url=new URL(process.env.DRIVERSI_LIVE_URL);
 assert.equal(url.protocol,'http:');assert.ok(['localhost','127.0.0.1'].includes(url.hostname));
 const browser=await chromium.launch({headless:true,executablePath:process.env.DRIVERSI_BROWSER_EXECUTABLE||undefined});
 try {
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(url.href);await page.waitForFunction(()=>state!==null);
  assert.equal(await page.evaluate(()=>LIVE),true);
  await page.locator('#pause').click();await page.locator('#reset').click();
  await page.waitForFunction(()=>state.world_tick===0&&!commandBusy);
  const bindings=await page.evaluate(()=>state.behavior_policies);
  assert.ok(Object.keys(bindings).length>=2);assert.ok(new Set(Object.values(bindings).map(b=>b.policy_id)).size>=2);
  await page.locator('#actor').selectOption('ego');
  assert.equal(JSON.parse(await page.locator('#selected').innerText()).policy_binding.policy_id,bindings.ego.policy_id);
  await page.locator('#step').click();await page.waitForFunction(()=>state.world_tick===1&&!commandBusy);
  await page.locator('#brake').click();await page.waitForFunction(()=>!commandBusy);
  await page.locator('#step').click();await page.waitForFunction(()=>state.world_tick===2&&!commandBusy);
  assert.equal(await page.evaluate(()=>state.latest_transition.action_origins.ego),'override');
  assert.equal(await page.evaluate(()=>state.latest_transition.executed_actions.ego),1);
  await page.locator('#play').click();await page.waitForFunction(()=>state.world_tick>2);
  await page.locator('#pause').click();await page.waitForFunction(()=>!commandBusy);
  const tick=await page.evaluate(()=>state.world_tick);await page.waitForTimeout(600);
  assert.equal(await page.evaluate(()=>state.world_tick),tick);
  await page.locator('#reset').click();await page.waitForFunction(()=>state.world_tick===0&&!commandBusy);
  assert.deepEqual(await page.evaluate(()=>state.behavior_policies),bindings);
  assert.deepEqual(errors,[]);assert.equal(await page.locator('#error').innerText(),'');
 } finally {await browser.close();}
});
