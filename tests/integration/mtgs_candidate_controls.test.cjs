const {test}=require('node:test');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.DRIVERSI_PLAYWRIGHT_MODULE||'playwright');

test('candidate channel, complete action interval, timeline and source deep link', async()=>{
 const url=new URL(process.env.DRIVERSI_CANDIDATE_URL);
 assert.equal(url.protocol,'http:');assert.ok(['localhost','127.0.0.1'].includes(url.hostname));
 const browser=await chromium.launch({headless:true,executablePath:process.env.DRIVERSI_BROWSER_EXECUTABLE||undefined});
 try {
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(url.href);await page.waitForFunction(()=>clip!==null);
  assert.equal(await page.locator('#channel').inputValue(),'interaction_intervention');
  assert.equal(await page.evaluate(()=>clip.trainable),false);
  for(const img of await page.locator('#images img').all()) await img.evaluate(i=>i.decode());
  const last=Number(await page.locator('#scrub').getAttribute('max'));
  await page.locator('#scrub').fill(String(last));await page.waitForFunction(i=>index===i,last);
  await page.locator('#pause').click();assert.equal(await page.evaluate(()=>index),last);
  const interval=JSON.parse(await page.locator('#actions').innerText());assert.equal(interval.joint_actions.length,interval.to_tick-interval.from_tick);
  const target=await page.locator('#full').getAttribute('href');const expected=await page.evaluate(()=>({generation:selected.generation,tick:clip.sensors[index].tick}));
  const replay=await browser.newPage();await replay.goto(new URL(target,url).href);
  assert.deepEqual(await replay.evaluate(()=>({generation:state.generation,tick:state.tick})),expected);
  assert.ok(Object.keys(await replay.evaluate(()=>state.behavior_policies)).length>=2);
  await page.locator('#channel').selectOption('render_quality');await page.waitForFunction(()=>clip!==null&&selected.channel==='render_quality');
  assert.match(await page.locator('#info').innerText(),/渲染支持缺口/);
  await page.locator('#channel').selectOption('driving_event');
  const count=await page.locator('#cases option').count();
  if(count===0){assert.equal(await page.locator('#images img').count(),0);assert.equal(await page.locator('#time').innerText(),'');assert.ok(await page.locator('#scrub').isDisabled());}
  assert.equal(await page.locator('#error').innerText(),'');assert.deepEqual(errors,[]);
 } finally{await browser.close();}
});
