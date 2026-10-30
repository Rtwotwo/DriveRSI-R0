"""Publish prepared H3 conditioning samples only; never run video generation."""
import argparse
import html
import json
from pathlib import Path
import re
import shutil
import zipfile

from driversi.rendering.whitebox.delivery import MEMBERS
from driversi.rendering.whitebox.exporter import sha256

STYLE = '''<style>body{max-width:1250px;margin:30px auto;padding:0 20px;font:16px/1.6 system-ui;background:#eef3f7;color:#183249}h1{font-size:30px}a{color:#006d71}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}article{padding:20px;background:white;border-radius:12px}video{width:100%;background:#162b3a;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace;max-height:580px;overflow:auto}.note{background:#fff2d9;padding:16px;border-radius:9px}.button{display:inline-block;padding:9px 13px;background:#087b80;color:white;text-decoration:none;border-radius:6px;margin:5px 4px 5px 0}details{margin-top:15px}</style>'''


def page(title, body):
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title>' + STYLE + '<body>' + body + '</body></html>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text()); cards = []; deliveries = []
    for row in report['rows']:
        token = row['token']
        if not re.fullmatch('[0-9a-f]{16}', token): raise ValueError('Invalid token')
        if row['status'] not in ('completed', 'reused'):
            cards.append('<article><h2>' + token + '</h2><p>' + html.escape(row['status']) + '</p></article>'); continue
        if row.get('prompt_profile') != 'minimax_h3_ref2va_v1': raise ValueError('H3 receipt required')
        archive = Path(row['archive'])
        if sha256(archive) != row['archive_sha256']: raise ValueError('Changed archive')
        target = args.output / token; target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            if sorted(z.namelist()) != sorted(MEMBERS) or z.testzip(): raise ValueError('Wrong five-file package')
            for name in MEMBERS: (target / name).write_bytes(z.read(name))
        shutil.copyfile(archive, target / archive.name)
        document = json.loads(Path(row['h3_condition']).read_text())
        source = json.loads((Path(row['bundle']) / 'source.json').read_text())
        city = source['map_name']; panels = []
        for job in document['jobs']:
            cam = job['camera_id']
            panels.append('<article><h2>' + cam + ' · ' + html.escape(job['facing']) + '</h2>'
                          '<video controls playsinline preload="metadata" src="' + cam + '.mp4"></video>'
                          '<a download href="' + cam + '.mp4">单独下载白膜视频</a>'
                          '<details><summary>该视角的六节原生prompt</summary><pre>' + html.escape(job['request']['prompt']) + '</pre></details></article>')
        intro = '<a href="../index.html">返回场景列表</a><h1>' + html.escape(city) + ' · ' + token + '</h1>'
        intro += '<p class="note">这里播放的是原始白膜输入，尚未运行H3生成。输入7秒/10fps；H3原生目标7秒/24fps。每次仅给一个相机视频，绑定为&lt;Video 1&gt;，使用下方该相机的prompt。</p>'
        intro += '<a class="button" download href="whitebox-7s10hz-with-prompts.zip">下载四路MP4＋一个TXT</a><a class="button" download href="video-generation-prompt.txt">查看／下载H3提示词TXT</a>'
        intro += '<p>TXT汇总四个独立任务，不要整份提交一次。JSON请求保存在服务器exp目录，不混进下载包。四路输出的外观一致性仍需生成后检查。</p>'
        (target / 'index.html').write_text(page(city + ' H3条件', intro + '<div class="grid">' + ''.join(panels) + '</div>'), encoding='utf-8')
        cards.append('<article><h2>' + html.escape(city) + '</h2><p>' + token + '</p><video controls playsinline preload="metadata" src="' + token + '/CAM_F0.mp4"></video><a class="button" href="' + token + '/index.html">四视角与对应prompt</a><a download href="' + token + '/whitebox-7s10hz-with-prompts.zip">下载五文件包</a></article>')
        deliveries.append((token, archive))
    combined = args.output / 'all-h3-whitebox-samples.zip'
    temporary = combined.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as out:
        for token, archive in deliveries:
            with zipfile.ZipFile(archive) as src:
                for name in MEMBERS: out.writestr(token + '/' + name, src.read(name))
    with zipfile.ZipFile(temporary) as z:
        if len(z.namelist()) != 5 * len(deliveries) or z.testzip(): raise ValueError('Combined archive failed')
    temporary.replace(combined)
    body = '<h1>MiniMax-H3 · 四相机白膜条件与专用提示词</h1><p>原版样例保留；本页提供官方本地Ref2VA格式的新版本。</p><p class="note">这是H3输入条件演示，不是H3生成视频。' + str(len(deliveries)) + '份已审阅场景 → ' + str(4*len(deliveries)) + '个独立相机任务。六节英文prompt与引用绑定已验证；画面生成与跨视角一致性尚未验证。</p>'
    body += '<a class="button" download href="' + combined.name + '">下载全部H3条件样例</a><a href="../caption-whitebox-multicity-v1/index.html">原版通用提示词样例</a>'
    body += '<p>每个场景只包含四个MP4和一个TXT，合集没有JSON、PNG、拼屏或额外说明文件。</p><div class="grid">' + ''.join(cards) + '</div>'
    body += '<p>开源H3许可第V.3条限制使用H3输出改进其他AI模型；用于ChainFlow训练前需确认适用授权。此处仅交付原始白膜输入与提示词。</p>'
    (args.output / 'index.html').write_text(page('MiniMax-H3白膜条件样例', body), encoding='utf-8')
    print(json.dumps(dict(scenes=len(deliveries), archive=str(combined), members=5*len(deliveries), bytes=combined.stat().st_size)))


if __name__ == '__main__': main()
