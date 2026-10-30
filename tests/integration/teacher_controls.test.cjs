const {test}=require('node:test');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.DRIVERSI_PLAYWRIGHT_MODULE||'playwright');

test('teacher report and learned branches expose results and usable controls',async()=>{
 const base=new URL(process.env.DRIVERSI_TEACHER_URL);
 assert.equal(base.protocol,'http:');assert.ok(['127.0.0.1','localhost'].includes(base.hostname));
 const browser=await chromium.launch({headless:true,executablePath:process.env.DRIVERSI_BROWSER_EXECUTABLE||undefined});
 try{
  const page=await browser.newPage({viewport:{width:1300,height:900}}),errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(new URL('index.html',base).href);
  assert.equal(await page.locator('tbody tr').count(),3);
  await page.locator('#metric').selectOption('entropy');
  assert.match(await page.locator('#range').innerText(),/152/);
  const links=await page.locator('nav a').evaluateAll(xs=>xs.map(x=>x.href));
  assert.equal(links.length,3);
  for(const url of links){
   await page.goto(url);await page.waitForFunction(()=>state!==null);
   assert.equal(await page.evaluate(()=>state.sampling.decision_mode),'privileged_structured');
   assert.equal(await page.evaluate(()=>CONFIG.manifest.protocol.rgb_is_policy_input),false);
   assert.equal(await page.evaluate(()=>CONFIG.manifest.sampling.policy.approved_teacher),false);
   await page.locator('#scrub').fill('6');await page.waitForFunction(()=>recordIndex===6);
   assert.equal(await page.evaluate(()=>state.tick),12);
   await page.locator('#actor').selectOption('2903abf058475d78');
   for(const img of await page.locator('img.camera-image').all())await img.evaluate(i=>i.decode());
   await page.locator('#step').click();await page.waitForFunction(()=>state.tick===14);
   await page.locator('#previous').click();await page.waitForFunction(()=>state.tick===12);
   assert.equal(await page.locator('#error').innerText(),'');
  }
  assert.deepEqual(errors,[]);
 }finally{await browser.close();}
});
