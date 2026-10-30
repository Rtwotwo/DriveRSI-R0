"""Local-only caption review page. Never publishes or calls providers."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from driversi.data.catalog.bailian_caption import write

HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriveRSI · 四城市结构绑定描述</title><style>
body{font:16px/1.65 system-ui;margin:0;background:#eff4f7;color:#163047}header,main{max-width:1360px;margin:auto;padding:24px}header{background:#102c42;color:white;max-width:none}header>div{max-width:1360px;margin:auto}h1{font-size:27px;margin:0}section{background:white;border:1px solid #d8e1e8;border-radius:12px;padding:20px;margin:16px 0}button,select,input{padding:10px;border:1px solid #bccbd7;border-radius:7px;background:white;color:#163047}input{width:330px}button{cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.grid img{width:100%;border-radius:6px}.muted{color:#596d7f}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:14px;border-radius:7px;font:14px/1.6 ui-monospace}a{color:#077a84}label{display:inline-block;margin-right:12px}.badge{background:#fff1d6;padding:10px;border-radius:6px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dde5eb;padding:7px;text-align:left}#numbers{display:flex;gap:25px;flex-wrap:wrap}.claim{border-left:3px solid #19a09d;padding:8px 14px;margin:8px 0}.bad{border-color:#c56b22}@media(max-width:760px){.grid{grid-template-columns:1fr}input{width:90%}}
</style><header><div><h1>DriveRSI · 几何事实与视觉描述分离</h1><p>64 个 navtrain 场景 · 64 个原始来源组 · 四城市 · 白膜决定位置与运动，VLM 补充外观候选</p></div></header><main>
<section><div id="numbers"></div><p class="badge">这是标注与提示词质量试验，未运行视频生成。文字草稿不等于视觉准入；本批几何文件不含道路面与完整遮挡条件，不能单独作为完整生成条件。</p><a href="analysis.json" download>下载完整分析</a> · <a href="selection.json" download>64场景清单</a> · <a href="retrieval-protocol.json" download>检索验证口径</a></section>
<section><label>浏览筛选（非语义检索） <input id="filter" placeholder="城市、token 或描述关键词"></label><select id="scene"></select><select id="scope"><option value="retrieval_history">检索：历史至锚点</option><option value="video_clip">视频：完整7秒片段</option></select><p id="identity"></p><div class="grid" id="photos"></div><p class="muted">展示锚点实拍四相机；VLM 实际读取每范围四个时刻 × 四相机。照片不是生成结果。</p></section>
<section><h2>外观候选与证据</h2><p id="state"></p><div id="claims"></div><p><a id="captionDownload" download>下载描述JSON</a> · <a id="geometryDownload" download>下载当前范围白膜轨迹JSON</a></p></section>
<section><h2>锚点对象：直接来自白膜</h2><p class="muted">固定锚点自车坐标：x前、y左、z上。朝向为同一坐标系角度；位移来自标注，不代表实测速度或行为意图。0.1秒为离线插值。</p><table><thead><tr><th>ID</th><th>类别</th><th>xyz / m</th><th>朝向 / °</th><th>采样依据</th></tr></thead><tbody id="actors"></tbody></table></section>
<section><h2>提示词预览</h2><p class="badge">含未审核外观的草稿。正式 released_prompts 仍只含几何；外观需按证据复核后发布。</p><pre id="prompt"></pre></section>
<section><h2>检索与成本</h2><pre id="metrics"></pre><p>结构检索使用同一套来源谓词作相关性标签，只验证接口连通性。外观检索仅基于固定16场景的助手图片复核，不能外推为全量准确率。</p></section>
</main><script>
let db,filtered,current,revision=0;const el=id=>document.getElementById(id);function txt(tag,value,cls){let n=document.createElement(tag);n.textContent=value;if(cls)n.className=cls;return n}
async function show(){const rev=++revision;el('claims').replaceChildren();el('actors').replaceChildren();el('prompt').textContent='加载当前范围…';el('state').textContent='加载当前范围…';el('captionDownload').removeAttribute('href');el('geometryDownload').removeAttribute('href');let row=db.scenes.find(x=>x.token===el('scene').value);if(!row)return;current=row;let scope=el('scope').value;let ref=row.scopes[scope];el('identity').textContent=row.city+' · '+row.token+' · '+row.source_group;el('photos').replaceChildren();for(let p of row.photos){let d=document.createElement('div');d.append(txt('div',p.camera));let im=new Image();im.src=p.file;im.alt=p.camera+'真实锚点照片';d.append(im);el('photos').append(d)}
let [r,g]=await Promise.all([fetch(ref.caption).then(x=>x.json()),fetch(ref.geometry).then(x=>x.json())]);if(rev!==revision)return;el('state').textContent='状态：'+r.status+'；自动外观批准：否';el('claims').replaceChildren();for(let [k,c]of Object.entries(r.candidate_claims||{})){el('claims').append(txt('div',k+' — '+c.text+'\n证据：'+c.evidence_ids.join(', '),'claim'))}for(let c of r.quarantined_claims||[]){el('claims').append(txt('div','隔离 '+c.path+' — '+c.claim.text,'claim bad'))}if(r.error)el('claims').append(txt('pre',r.error));el('actors').replaceChildren();for(let a of g.actor_summaries.filter(x=>x.at_anchor).slice(0,12)){let tr=document.createElement('tr');for(let v of [a.actor_id,a.semantic_class,JSON.stringify(a.at_anchor.position_in_anchor_ego_m),a.at_anchor.heading_in_anchor_ego_degrees,a.at_anchor.presence_evidence])tr.append(txt('td',String(v)));el('actors').append(tr)}el('captionDownload').href=ref.caption;el('geometryDownload').href=ref.geometry;el('prompt').textContent=r.draft_prompts?.retrieval_text||r.draft_prompts?.video_positive_prompt||'未形成合格格式的草稿';}
function filter(){let q=el('filter').value.toLowerCase();filtered=db.scenes.filter(x=>(x.city+' '+x.token+' '+x.search).toLowerCase().includes(q));el('scene').replaceChildren();for(let s of filtered){let o=txt('option',s.city+' · '+s.token);o.value=s.token;el('scene').append(o)}show()}
fetch('site-data.json').then(x=>x.json()).then(d=>{db=d;let a=d.analysis;for(let t of ['场景 '+a.scene_count,'格式合格 '+a.final_format_valid+'/'+a.final_scope_records,'请求 '+a.api_requests_including_spine,'忙时估算 ¥'+a.published_price_estimate_rmb.busy.toFixed(3)])el('numbers').append(txt('strong',t));el('metrics').textContent=JSON.stringify({appearance_retrieval_16_audit:a.appearance_retrieval_16_audit,diversity:a.appearance_text_diversity,cost:a.published_price_estimate_rmb},null,2);filter()});el('filter').oninput=filter;el('scene').onchange=show;el('scope').onchange=show;
</script></html>'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    e=Path(a.experiment);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    selected=json.loads((e/'selection.json').read_text());analysis=json.loads((e/'analysis.json').read_text())
    jobs=json.loads((e/'jobs.json').read_text());jobmap={(j['token'],j['scope']):j for j in jobs}
    rows=json.loads((e/'pilot-v3_1/report.json').read_text())['rows'];results={(r['token'],r['scope']):r for r in rows}
    scenes=[]
    for row in selected['rows']:
        token=row['token'];item={k:row[k]for k in ['token','city','source_group']};item.update(scopes={},photos=[],search='')
        for scope in ['retrieval_history','video_clip']:
            job=jobmap[token,scope];result=results[token,scope];r=json.loads(Path(result['file']).read_text())
            dst=out/'data'/token;dst.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(job['geometry_file'],dst/(scope+'-geometry.json'))
            # Remove runtime connection fingerprints from the downloadable caption.
            r.pop('profile',None);write(dst/(scope+'-caption.json'),r)
            item['scopes'][scope]={'caption':f'data/{token}/{scope}-caption.json','geometry':f'data/{token}/{scope}-geometry.json'}
            if scope=='retrieval_history':
                item['search']=' '.join(c['text']for c in r.get('candidate_claims',{}).values())
                for image in job['evidence']:
                    if image['id'].endswith('_f+0'):
                        name=image['sha256']+'.jpg';(out/'images').mkdir(exist_ok=True)
                        shutil.copyfile(image['file'],out/'images'/name)
                        item['photos'].append({'camera':image['camera'],'file':'images/'+name})
        scenes.append(item)
    write(out/'site-data.json',{'scenes':scenes,'analysis':analysis})
    for name in ['analysis.json','selection.json','retrieval-protocol.json']:shutil.copyfile(e/name,out/name)
    (out/'index.html').write_text(HTML,encoding='utf-8')
    print(json.dumps({'scenes':len(scenes),'page':str(out/'index.html')}))


if __name__=='__main__':main()
