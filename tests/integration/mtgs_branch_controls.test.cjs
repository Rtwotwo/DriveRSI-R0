const {test}=require('node:test');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.DRIVERSI_PLAYWRIGHT_MODULE||'playwright');

test('checkpoint galleries retain lineage, timestamps and usable camera controls',async()=>{
 const base=new URL(process.env.DRIVERSI_BRANCH_URL);
 assert.equal(base.protocol,'http:');assert.ok(['127.0.0.1','localhost'].includes(base.hostname));
 const browser=await chromium.launch({headless:true,executablePath:process.env.DRIVERSI_BROWSER_EXECUTABLE||undefined});
 try{
  for(const mode of ['structured','visual'])for(const branch of ['parent','replay','brake']){
   const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
   await page.goto(new URL(mode+'/'+branch+'/viewer.html',base).href);
   await page.waitForFunction(()=>state!==null);
   assert.equal(await page.evaluate(()=>state.sampling.decision_mode),mode);
   assert.equal(await page.evaluate(()=>CONFIG.manifest.trainable),false);
   if(branch!=='parent'){
    assert.equal(await page.evaluate(()=>CONFIG.manifest.branch_lineage.at(-1).tick),11);
    assert.equal(await page.evaluate(()=>CONFIG.manifest.branch_lineage.at(-1).prefix_is_new_sample),false);
   }
   await page.locator('#scrub').fill('6');await page.waitForFunction(()=>recordIndex===6);
   assert.equal(await page.evaluate(()=>state.tick),12);
   assert.ok(Math.abs(await page.evaluate(()=>state.time_seconds)-1.2)<1e-9);
   await page.locator('#actor').selectOption('2903abf058475d78');
   for(const img of await page.locator('img.camera-image').all())await img.evaluate(i=>i.decode());
   await page.locator('#step').click();await page.waitForFunction(()=>state.tick===14);
   await page.locator('#previous').click();await page.waitForFunction(()=>state.tick===12);
   assert.deepEqual(errors,[]);assert.equal(await page.locator('#error').innerText(),'');
   await page.close();
  }
 }finally{await browser.close();}
});
