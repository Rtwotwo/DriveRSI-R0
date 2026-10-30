const {test}=require('node:test');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.DRIVERSI_PLAYWRIGHT_MODULE||'playwright');

test('structured and visual recorded modes expose actual input and held decisions', async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.DRIVERSI_BROWSER_EXECUTABLE||undefined});
 try {
  for(const [mode,url] of [['structured',process.env.DRIVERSI_STRUCTURED_URL],['visual',process.env.DRIVERSI_VISUAL_URL]]){
   const target=new URL(url);assert.equal(target.protocol,'http:');assert.ok(['127.0.0.1','localhost'].includes(target.hostname));
   const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
   await page.goto(url);await page.waitForFunction(()=>state!==null);
   assert.equal(await page.evaluate(()=>state.sampling.decision_mode),mode);
   const info=JSON.parse(await page.locator('#sampling').innerText());assert.equal(info.mode,mode);
   assert.equal(await page.evaluate(()=>CONFIG.manifest.protocol.rgb_is_policy_input),mode==='visual');
   if(mode==='visual'){
    assert.equal(await page.evaluate(()=>state.sampling.visual_actor_ids.length),2);
    await page.locator('#generation').selectOption('1');
    await page.locator('#scrub').fill('1');await page.waitForFunction(()=>recordIndex===1);
    const dec=await page.evaluate(()=>state.transition.policy_decision);
    assert.equal(dec.observation_tick,0);assert.equal(dec.newly_decided,false);
    assert.equal(dec.input_mode,'visual');
    assert.match(await page.locator('#sampling').innerText(),/diagnostic_not_learned_or_teacher/);
   }
   for(const img of await page.locator('img.camera-image').all())await img.evaluate(i=>i.decode());
   assert.deepEqual(errors,[]);assert.equal(await page.locator('#error').innerText(),'');await page.close();
  }
 }finally{await browser.close();}
});
