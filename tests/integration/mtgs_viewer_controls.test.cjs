/* Explicit browser integration: serve a completed MTGS gallery on loopback.
 * Set DRIVERSI_VIEWER_URL, optionally DRIVERSI_PLAYWRIGHT_MODULE and
 * DRIVERSI_BROWSER_EXECUTABLE. No GPU or world-control requests are made.
 */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.DRIVERSI_PLAYWRIGHT_MODULE || 'playwright');

test('recorded timeline preserves a requested seek across pause and branch reset', async () => {
  const url = new URL(process.env.DRIVERSI_VIEWER_URL);
  assert.equal(url.protocol, 'http:');
  assert.ok(['127.0.0.1', 'localhost'].includes(url.hostname));
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.DRIVERSI_BROWSER_EXECUTABLE || undefined});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(url.href);
    assert.equal(await page.evaluate(() => LIVE), false);
    const groups = await page.locator('#generation option').count();
    assert.ok(groups >= 2, 'requires a paired gallery');
    await page.locator('#generation').selectOption('1');
    const last = Number(await page.locator('#scrub').getAttribute('max'));
    assert.ok(last >= 2);
    await page.locator('#scrub').fill(String(last));
    await page.waitForFunction(index => recordIndex === index, last, {timeout: 5000});
    assert.equal(await page.locator('#play').isDisabled(), true);
    await page.locator('#scrub').fill('1');
    await page.waitForFunction(() => recordIndex === 1, undefined, {timeout: 5000});
    assert.equal(await page.locator('#play').isDisabled(), false);
    await page.locator('#generation').selectOption('0');
    assert.equal(await page.locator('#scrub').inputValue(), '0');
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});

test('virtual owner cameras never link ego source photographs', async () => {
  const url = new URL(process.env.DRIVERSI_VIEWER_URL);
  assert.equal(url.protocol, 'http:');
  assert.ok(['127.0.0.1', 'localhost'].includes(url.hostname));
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.DRIVERSI_BROWSER_EXECUTABLE || undefined});
  try {
    const page = await browser.newPage();
    await page.goto(url.href);
    const virtualIds = await page.evaluate(() => state.cameras.filter(c => c.virtual).map(c => c.camera_id));
    assert.ok(virtualIds.length > 0, 'requires a gallery with virtual owner cameras');
    for (const id of virtualIds) {
      await page.locator('#camera-selection label').filter({hasText: id}).locator('input').check();
      const card = page.locator('#cameras article').filter({has: page.locator('h2', {hasText: id})});
      assert.equal(await card.locator('a').filter({hasText: '采集照片'}).count(), 0);
      await card.locator('img.camera-image').evaluate(image => image.decode());
      assert.ok(await card.locator('img.camera-image').evaluate(image => image.naturalWidth > 0));
      assert.match(await card.innerText(), /新视角质量未认证/);
    }
  } finally {
    await browser.close();
  }
});


test('height proxy and unsupported scene warnings remain distinct at the last frame', async t => {
  const url = new URL(process.env.DRIVERSI_VIEWER_URL);
  assert.equal(url.protocol, 'http:');
  assert.ok(['127.0.0.1', 'localhost'].includes(url.hostname));
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.DRIVERSI_BROWSER_EXECUTABLE || undefined});
  try {
    const page = await browser.newPage();
    await page.goto(url.href);
    const mode = await page.evaluate(() => state.cameras[0]?.height_binding?.mode);
    if (mode !== 'reference-track') { t.skip('requires the bounded height gallery'); return; }
    await page.locator('#generation').selectOption('1');
    const last = Number(await page.locator('#scrub').getAttribute('max'));
    await page.locator('#scrub').fill(String(last));
    await page.waitForFunction(index => recordIndex === index, last);
    const views = await page.evaluate(() => state.cameras.map(c => ({
      id:c.camera_id, supported:c.height_binding?.supported, scene:c.scene_height_supported,
      delta:c.height_binding?.correction_m})));
    for (const view of views) {
      await page.locator('#camera-selection label').filter({hasText:view.id}).locator('input').check();
      const card = page.locator('#cameras article').filter({has:page.locator('h2', {hasText:view.id})});
      const text = await card.innerText();
      assert.match(text, /有限参考轨迹高程代理/);
      assert.ok(text.includes((view.delta*100).toFixed(2)));
      if (view.supported) assert.match(text, /不是图像质量验收通过/);
      else assert.match(text, /高程查询不支持/);
      if (view.scene === false) assert.match(text, /世界车辆高程未全部支持/);
    }
    assert.ok(views.some(c => c.supported && c.scene === false), 'needs a partially supported frame');
  } finally { await browser.close(); }
});
