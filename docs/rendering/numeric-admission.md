# Experimental numerical admission

`experimental_numeric25_v1` is the operator-authorized production experiment. It is separate from independently reviewed visual admission; old review failures are never rewritten.

Required evidence: actual 70 DriveRL actions / 71 states (0–7s), changed continuous history and exact replay, native counterfactual PDMS strictly >95 for the native four-second scoring horizon, mean canonical source reconstruction PSNR strictly >25dB recomputed from sealed images, and both seven-second and five-second-future ADE >=0.5m against retained family trajectories. Scene/source, ego motion, camera coverage, all image hashes and four-history/ten-future export remain validated. Source PSNR is not ground-truth novel-view quality.

Terrain, other-object appearance and VLM reviews are diagnostics. The explicit numeric receipt does not assert independent pixel review. AUGUstus joint targets use actual measured branch objects and their measured future positions, omit unavailable identities/projections and preserve those omissions. Geometry-only targets have unverified visibility; they are not a replication of released AutoDrive_P3 language annotations. Native joint loss is unchanged.

AUGUstus released SFT preprocessing merges additional AutoDrive_P3 solutions and reasoning traces with NAVSIM. The released configuration trains the prediction component from `solution_prediction`; raw NAVSIM logs alone do not contain that full language supervision. ChainFlow stage1 is the no-language alternative; its released stage1 YAML has `use_diffusion_decoder: false` and `train_diffusion_decoder_only: false`. Do not enable DiT or remove AUGUstus language loss to masquerade as the same model.

Data: dataset/synthetic_navsim/samples/<content-id> contains separate history/future sensors, logs and window metadata. exp contains admission, feature, training and evaluation evidence. Fresh generated-history features are encoded before an admitted index is added. Count unique currently validated branches/packages, not index-receipt count. Source-code changes may supersede old sample receipts; preserve them and publish a new attempt.

Production supports sealed completed-generation reuse only when source, seed, physical controls and asset match. Two >25dB asset-supported anchors target ten retained samples each, at most 64 attempts per anchor. A shortage is not filled with repeated trajectories. Existing full-training retrieval and full-test evaluation remain isolated; no test pixels enter training.


## Versioned24dB/0.3m expansion

The operator confirmed `experimental_numeric24_ade03_v1`: source-view PSNR >=24dB and both full7s/future5s ADE >=0.3m. Native counterfactual PDMS remains strictly >95; complete actual7s, current/previous trajectory hashes, camera-motion and source-image binding remain checked. This is a low-fidelity numerical experiment, not independent multiview validation. The original `experimental_numeric25_v1` still requires >25dB and0.5m; prior rejected receipts are never edited. New policy runs may reuse hashed completed physical/render payloads but publish separate admission and native supervision receipts.

ChainFlow cohorts may explicitly reserve up to128 samples/512 render attempts, bounded further by campaign config. The seven-anchor expansion reserves70 samples and140 renders,64 attempts per anchor. Actual shortage remains visible; no duplication or unexecuted points may fill quota. Both native training arms use an identical declared scheduler and update budget; fresh-feature and positive-learning-rate exposure gates still precede full navtest evaluation.
