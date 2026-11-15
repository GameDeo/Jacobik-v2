# A4t report — 2026-11-02

## Status

Completed the Amendment 1 analysis and subsequent stereo reclassifications locally from stored relaxations. Amendment 2 results below are retained as the audit history; the Amendment 3 section at the end supersedes their class counts and remaining-Z table. No new relaxations or VM work were run; both VMs remain down.

## Provenance and completed run

- Worktree branch: `jgen/a4-baseline`. VM bundle commit: `57358ee1713c801ac7454054942464c4601511c3`; production fingerprint: `273f2ccbf892`.
- A4b `units.csv` SHA-256: `376282b30b39be8eaf70841fd63a3e1dcf4528d7b9334bbb6a60c94b75d98e9f`.
- Ordered digest of the 105 selected pool manifests (code, seed, per-manifest SHA-256): `3ec65553ec614b03c6f18b98a77c5bc91d3404196e40e173e887db4b91513444`.
- 3,516 rows across 105 manifests: 3,188 new xTB relaxations, 328 A4b relaxations reused, and 35 reference re-relaxations for parity. Fetched output had 105 unit JSONs, all 3,516 rows, and 35 references; row counts matched the manifests.
- Final row outcomes: 3,516 `ok`, 0 capped, 0 failed. Reference energy parity: all 35 within 1e-3 kJ/mol; maximum absolute difference 0.0 kJ/mol.
- xTB individual-job wall sum: 20,238.5 s; CUDA MACE wall: 135.2 s. These are summed worker/model times, not elapsed VM wall time; elapsed wall time was not captured.
- Colab balance 483.18 before and 478.58 after; **4.60 units spent** (6-unit task cap). Two launch/setup attempts and one input-extraction launch failed and were stopped; their spend is included. Both VMs were fetched/count-checked before shutdown; active assignments are 0.
- Input digest details and run outputs are in the gitignored `research/data/jgen/campaign/A4t/` directory.

## Pool manifest SHA-256 values

Each digest below is the SHA-256 of the corresponding A4b pool manifest JSON used for row selection. The ordered combined digest is recorded in provenance.

| Parent | Seed | Manifest SHA-256 |
|---|---:|---|
| BABNAV | 11 | `472e8b91bc926d441a5f87b6812cd6dc0c2df9e8d87a69c846cdac4e0a489f1b` |
| BABNAV | 12 | `b7ab73dc20979413690b6a565f9eef65392c80f7debc1a52c0c296978c809639` |
| BABNAV | 13 | `8ab956c6a2f3aec1cff5b1145151ad0953248efffdb87fa7ba2aad470e64afd2` |
| BOCRAN | 11 | `8fb52cf06467a0c2fb0b5ee44e727473d17731c67190f80dee6b02c82fd25bb6` |
| BOCRAN | 12 | `878428301321ea58910a3277167927d05d418b8762c9fa001e9ee56ee60bf04e` |
| BOCRAN | 13 | `1542df8baeac2b8ad41582b01163b4869d80580cc8e7ed075d81181bc3abd435` |
| CAQLEP | 11 | `1da96af489244333453d299f83385c7bc9b1ab3fd41df7b26148aa0788ce251e` |
| CAQLEP | 12 | `bc4148b34190fd7157f8bd35c97577169996147d49906ae7d1c28d5d0e06de9a` |
| CAQLEP | 13 | `ff141f7d655286a4224ab98c944cd9b9ceea08fdd7d2c97d77b293b0aca86952` |
| CIFZEZ | 11 | `205b6685f949c22ab70b1966d8736c822637dee19c7faaa3ae5fda9feb8e1d7f` |
| CIFZEZ | 12 | `cd971184df74bd58ca95542a8abaf1f9b18b63e59ddbb2915d269090f18c58de` |
| CIFZEZ | 13 | `0a047a28f7411f1b4fba3482149e93db2c7dc50ac67989ccb4aec157823c6c38` |
| CIXYUD | 11 | `9250232728056e5ec47d617fe7c6e680e4a8571d748c4e91b765b0aa194e5c40` |
| CIXYUD | 12 | `b0fd7bed39f908722e6e36fa817753ec3cd45927879e732cd9a4aa3f62f7cd86` |
| CIXYUD | 13 | `22eb6beaa920fbb9845387c8b163eafbdf0214995d3c3881c984f1790296c061` |
| DAMVAQ | 11 | `9a651aa141a29dad523d0e47ef4bad974f238c7d7f83ba998cbc17cc8b0a8318` |
| DAMVAQ | 12 | `94a63fb09723a989f8be34c290f012cd76b649198dbac30e4e812f35e971b661` |
| DAMVAQ | 13 | `f411e27aa8b3a6c68ffad57853d537ebefffb6f2ac99757abeb7769319878eba` |
| DOHTOM | 11 | `44963068fd837d0ab684849cbb4d6fa84a9f7cfad93d8c48e595d2f09b8423fe` |
| DOHTOM | 12 | `26efa507edc2d70bf237cdfa6d61d60114e3e8d5f5408175babd3b5f5ed6a73d` |
| DOHTOM | 13 | `b7ec59d5e2b9e99facab5e39e4c266ff4a835dbc3fd5bf6b25ad9c846c1451d1` |
| DUHHIX | 11 | `59c92d362f1cfd9405e7e02e756d5b32fa50b9d54ed9defee8af3c46c1d8baf1` |
| DUHHIX | 12 | `c7fa16ab2ce1aa85cdd813e8c3c7d1e070979688803ae44ad0740c535cadbd7e` |
| DUHHIX | 13 | `3f6a0270ae08daf4c796499c54b65a4068d8155ba689589ddb9c10128a98126b` |
| EFOXII | 11 | `a04d8244fbe1eb02712aa1b0ac74bc4aa7e2a9db3a08c50b08b9b4f0a0df0fb3` |
| EFOXII | 12 | `24ec21a4e45fc32bddc91a3a33881ce2944154165867acce68d72a7cb869688c` |
| EFOXII | 13 | `88d453cdbfe598f0745cf5087079bbb4a21cf500fc0c836fce817e320183b3e3` |
| EZIBUL | 11 | `f8bec2cd9d5291859f7e1324152264ec47b387c818c339ae33d81402a69b5fbf` |
| EZIBUL | 12 | `f2fa305ad6ab305975cb40da82bdb2b41acb030e587bf7540f1a704d8a6e149b` |
| EZIBUL | 13 | `ebf593a2a2a02b0923060a2aca82baa07bae3a13cf032d6cd811f0e2b74890b8` |
| FEKKAJ | 11 | `122090c1fd5c66d1c269fe740796c6725350fd52b8b21c8cf2f6aca61a2eaf80` |
| FEKKAJ | 12 | `a2e1d7b434f5031a4d06c017e60323a1c0d6534056fc77c7fc5653708c30439e` |
| FEKKAJ | 13 | `a16d57f66b9ad1e76195512c405e18acd6bdb731962548447c4c970203444d6c` |
| FETTAZ | 11 | `30bd6403bb6c7ea7ef5af66c258d477dfc37c43cda647fbff9087e76c8f23506` |
| FETTAZ | 12 | `fca371a9384d0a49bb5971fa898e2929863680f5f06073dae60b4cf5ef8229c5` |
| FETTAZ | 13 | `2b47b1b52ee77de05b17d4bac6a99efbf6d34349bfcc53e519a8010120a55e81` |
| FEXWUC | 11 | `5a78e47432397309bad50ce1f65d895c019226b3b2e71d714adcb33246b1d08d` |
| FEXWUC | 12 | `08278ce6d0ff42be40304a1690d9566ce03a5144ae51abd781697fc83a0bda9f` |
| FEXWUC | 13 | `7f64e9a53d5f13d37fc5f92a0cbcf86f9415c34683de94856c5cd9fcc6c189aa` |
| FUTSOD | 11 | `48d9db598b0006d7aa0c09de4a80335bfc2040d1c6635412ef4be6844e69dd00` |
| FUTSOD | 12 | `df5ddf5070389f68e7ee98246d61063d03a922f0d1a49b758d37ffc99b083c68` |
| FUTSOD | 13 | `1817ba5586e60c853a874893e04620d0a6ad5b421b91c00f23345d74e65fe908` |
| GOTWET | 11 | `91b9acef93f8d3fd79644c00d4ad1d1bc3947c5bbe895e3ba7f868f61036b678` |
| GOTWET | 12 | `15ed15a58648bf8984b007e3b95938267f2e10fd6a7a6066999f1f4a04fbade4` |
| GOTWET | 13 | `18ad0c58ede099099083b727cf81969825a7f10adc206beb29ff398fa812e831` |
| HOHZAG | 11 | `50802ba0b9aa01622bf857aa098b4e527cfbf9a1388a5b8acae39255b558dec9` |
| HOHZAG | 12 | `a5db050c6fce296e805236f058ca92db06d00c1ad8ac2282bea7fe8cab02390a` |
| HOHZAG | 13 | `327bc07cae21cf275efd7979ce2465df6daa91325356104778108c3f5b43905c` |
| HOXJUZ | 11 | `77e95f55bf8cffc7c94eb25e2a74fc7e318e951e77f3608b81d9b5d121c32bf4` |
| HOXJUZ | 12 | `018352a12c5f56e27e50535ad73ff0e02dfd1d70924633e964f9a2698629961a` |
| HOXJUZ | 13 | `9d4dd57f8de83fa02f5a1354f31b50945d4e044dff5247f51366e1f17faf5cab` |
| HOYBED | 11 | `c7570f09e319317be7133f96aff8925cdd711c931860d045aeb1fdcce98420b5` |
| HOYBED | 12 | `c110f3509b6c43ea8afb8cad066a59f92ccf7f0164f61b3db962d29e8d924e88` |
| HOYBED | 13 | `a46152a70199a99ef760d7f1f2036caee83a5ce3790ce801e75a5fefbd0c7340` |
| LIPDEU | 11 | `0a13cf1cd64540c4cb1ff387ed8f1d743b63df729f462c6e773355508cd5b49b` |
| LIPDEU | 12 | `61fab00d0c4af0978a0b974fe40a07d9e949d8074a356387b21b0a8caa8a0f87` |
| LIPDEU | 13 | `361448cd54dc07b984eefbc9787913c8ee94d57efd4ba738b09169ec7ff51d4e` |
| LUQVOK | 11 | `641e83bb9f8c994bd3c8120ca77fc67518a459e2abfe3204c0b53b93ea125c70` |
| LUQVOK | 12 | `4e19715e95befc06a6cea45400892f1400d9052a9795df40826786c512ec7a31` |
| LUQVOK | 13 | `c7ecb06082a861c448700cd03f172502e009b4a8c5c6e8136e02937bbaeec622` |
| MIDQEY | 11 | `4ecddc4e399c0af74503b9751db815232fce002674059c68b74b0a2f04dc92cb` |
| MIDQEY | 12 | `c66392557768e9f3eb45ac6bfda09ba3dcbd2909cf463ad8a9784e0c4dab0d25` |
| MIDQEY | 13 | `d2720d6b50ec5d2d061939539bdf8f4aefaedabf40fda9bf6056eba8f02237a5` |
| NUCKUS | 11 | `8107c2b8ba8ecbbb2a7c769280cd72ba6137cb20d370176ce89aff5928582e5d` |
| NUCKUS | 12 | `4dbd2d9d7113ac400e024db10f0f22780f5b5b09ef2db0993a17cb7a2b20fb4b` |
| NUCKUS | 13 | `c737d11f2670af9293b1834d7d6fd325ecca31249674392d358270a44db63154` |
| NUWFAM | 11 | `042a9e4f4c6905d3e2505d1bd0c839d1803bbb739efbc1872971f57ca5d8e802` |
| NUWFAM | 12 | `411ccf41a3397d8527791354562f8a67d14a0b4c31f2fab3100299f24b15ca5e` |
| NUWFAM | 13 | `117e6dc59c0fea53d6302b4fdb3f0a3ce35f80cfc5b2ad9e59421c790d812051` |
| QAPFAS | 11 | `8ddeff77f2c7834bc5fa07dd5333a10a2205605df6059a4949d8dfa4325ad9f5` |
| QAPFAS | 12 | `a5530d8ec559f3a18b2288cb939e37e144a8294271ee074b4255eb776ab62a8c` |
| QAPFAS | 13 | `d841c92693f99aadc476bf200e643638736d60a0dad8a75b98857597ae79bc36` |
| SILROY | 11 | `1556665605e1429fa9cb061ede7440e6ee49f7b40b56be361d3c18f49c464047` |
| SILROY | 12 | `4c9b321111059a2ddd18be3bbdeb7f7de90687cd935b77179c6bba4874513c4a` |
| SILROY | 13 | `4378ec031022b5837ef43d0c635653307b60dba5b65468eaeb93b627f89024fa` |
| SIVBUV | 11 | `fcb5f1445503bab281b65a61e5829d31632dbedd1caa83eadbb303ba7c531371` |
| SIVBUV | 12 | `8354566b1ccdf8bc7888799c9f26937c8681bb4bf55decfb9189bd42bd2d32ca` |
| SIVBUV | 13 | `a71888e39aacdd39828f0d3f480e6f4262a67d958c19bf3cba29e9c15b54b5f2` |
| USAPOU | 11 | `70973dbe30bf288a793a2d791443ee7c1df8be6bd01b5ae5a72b7f5ab4fc56db` |
| USAPOU | 12 | `3e337bea2214a3da8afeaf9795aedb6bd5f0f80a543536138826bbcc0a131788` |
| USAPOU | 13 | `0b8c7a9143c292f7e70270a0c818701905f5f417a1f63a3f5731a56e00426354` |
| UYOVUA | 11 | `b0132173af9333fbacf26586d0a0cd350814bc37a61994bb4b0ef34b66f50d98` |
| UYOVUA | 12 | `87e88c38fccd3d9b4d295885e943cfb42dc17bc57cb4f44ec7b794eec521f260` |
| UYOVUA | 13 | `97cfbcf5f1bcee8977fb6b214f97f99801cfed04c5a24c40e0fb95122a44b929` |
| XACMOE | 11 | `c5b45329269c1a2e3b3dc35819be3e813532e36d777d9d6e389e35ef9653c08e` |
| XACMOE | 12 | `392129eedee4c7fe9808f709e421a9bbd5b11d7db004ef108b7579c38bfe13d1` |
| XACMOE | 13 | `ed7cb4283d7f33bc64ea6502ec91af8025272bac99b0c53ccb7a3fb8cf2bcbf1` |
| XEVDAE | 11 | `284efae05fce762393f19972d83894461e915a798ce2bd036d35c08920d21fca` |
| XEVDAE | 12 | `2d8882e946146058321cf6645ecfd0de77b3c79d60e7a0ab9cfad4c1e3f5cd2c` |
| XEVDAE | 13 | `372c3f2f10ba25a7600900554f6273f293b9ee7f992e4f1b0cdcabcac8e301ea` |
| XOJHIN | 11 | `33d0160d8e9111345c153ab955aa0f67fbcf9ef1479eb7a5ebe9228895d487a5` |
| XOJHIN | 12 | `464114fd3ca9d5cc974ea61d82135dafa270942beae504b131d566ee925692dc` |
| XOJHIN | 13 | `de1861bddca3b49729bc0000e35a8b731e8b28cb1da769928084bf79d014c151` |
| YOVNEC | 11 | `271cb13ef979129e0149e0642d339bebb0a2e4db792112d63982bb7f347feb79` |
| YOVNEC | 12 | `1ba90468ed9e9180d93d745bcd0d38fb3ce4c89320d0a534fab5ffb617b3af23` |
| YOVNEC | 13 | `37942a260a4f229dad057d52caebaad358ccab9f99a769836a72dee4cb33898f` |
| ZAFPOP | 11 | `2b1795e9edd02f20d77d4e4ee71afec38b349184c3331fabbc98974a0b7f5f49` |
| ZAFPOP | 12 | `5095b22d627f428b9b80f9f35ca2401a27ee22dc443aee40e3edc1b5b7d17674` |
| ZAFPOP | 13 | `ddbd4962ff7ac3dd7b4dff2b30abb92e8b755befe205cad35a154d5e7094affb` |
| ZAQNUE | 11 | `15d1faf6105f1f9569155fc7bc4b75e409cc3f4e51135f17e3daa24573dca205` |
| ZAQNUE | 12 | `5fbc11912df3142babfee4263a35a2d1107025d048ab416fc99a19934b83e483` |
| ZAQNUE | 13 | `422c61d1a496d4ac716fdcf797f94ee738c5e330bcc6336fa9a7c421715cd603` |
| ZAZREZ | 11 | `63bffa51675f5f26d145416e01a717f3b91b0b7b9c8d9527679bf1097b40ee04` |
| ZAZREZ | 12 | `4ecce51ab8823c074ba693897ad1169bfcacfa971c13c8fc449d0f607f79eed0` |
| ZAZREZ | 13 | `aa092224a73108c79de79b0e88d23f9c9be7d95a22bab473034780eab71e4295` |

## Amendment 1 geometry and donor sensitivity

Reference geometry is undefined when relaxed-reference donor CN is not represented in `IDEAL_GEOMETRIES` (supported CN 2–9). No shape measure is calculated in that case. T is evaluated before G; if undefined geometry reaches the G position it is class `G*`. The analysed panel has no `G*`: GOTWET and XEVDAE have CN 11, and NUWFAM has CN 14; their ordered classes are T, T, and F respectively. GOTWET’s CN-11 donor elements are Si₃N₃O₃C₂ on Zr, as anticipated.

For the five parents classified T by donor-set mismatch, the builder donor set, pad-0.6 reference donor set and pad-0.3 reference donor set (zero-based atom indices) are:

| Parent | Builder donors | Ref pad 0.6 | Ref pad 0.3 | Mismatch remains at 0.3? | Reference CN / elements |
|---|---|---|---|---|---|
| DAMVAQ | `[1, 10, 15]` | `[1, 2, 8, 10, 11, 15]` | `[1, 2, 8, 10, 15]` | True | 6 / P,C,C,C,C,C |
| FUTSOD | `[1, 3, 5, 8, 25, 42]` | `[1, 2, 3, 5, 8, 25, 42]` | `[1, 3, 5, 8, 25, 42]` | False | 7 / S,P,O,S,N,N,N |
| GOTWET | `[4, 5, 6, 9, 34]` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 34, 74]` | `[4, 5, 6, 7, 9, 34]` | True | 11 / Si,Si,Si,N,N,N,O,O,O,C,C |
| NUCKUS | `[1, 33, 67, 101]` | `[1, 33, 34, 67, 68, 101, 102]` | `[1, 33, 67, 101]` | False | 7 / P,N,C,N,C,N,C |
| XEVDAE | `[2, 7, 13, 14, 15, 22, 23, 24]` | `[1, 2, 7, 11, 12, 13, 14, 15, 22, 23, 24]` | `[2, 7, 13, 14, 15, 22, 23, 24]` | False | 11 / S,O,C,C,C,C,C,C,C,C,C |

NUWFAM also has a donor-set mismatch at both pads but is class F because the F rule precedes T. It is included in the main parent table, not the T-only sensitivity table.

## Amendment 2 stereo reclassification

Stereo labels are compared after mapping the best shell-matched relaxed row onto the reference with the same heavy-atom RMSD-optimal automorphism used by `ring_share`. The metal label is excluded; reference double-bond labels must match; centre comparison uses labels present in both mapped structures and accepts them when they all match or all invert. Reclassification uses the same decision list, so parents leaving Z fall through to R or C. No new relaxations were run.

### Old versus new aggregate classes

| Rule set | F | T | G | G* | S | Z | R | C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amendment 1 (old) | 5 | 5 | 1 | 0 | 2 | 13 | 6 | 3 |
| Amendment 2 (new) | 5 | 5 | 1 | 0 | 2 | 6 | 9 | 7 |

### Causes for parents classified Z under Amendment 1

“Old differences” lists differing RDKit labels at metal, donor, other ligand-centre, and double-bond elements. A metal label found only on the row is shown as a difference for audit visibility; the earlier one-sided comparison did not itself count a row-only label. “Removed by” lists Amendment 2 changes that resolve one or more old differences. “Remaining” lists the mapped, mirror-tolerant differences that keep the parent in Z.

| Parent | New class | Old differing elements | Removed by Amendment 2 | Remaining difference |
|---|---|---|---|---|
| BABNAV | R | ligand centre | RMSD permutation | none |
| BOCRAN | Z | double bond, ligand centre | mirror tolerance | double bond |
| CAQLEP | C | ligand centre, metal | metal exclusion, RMSD permutation | none |
| CIFZEZ | Z | double bond | none | double bond |
| FEKKAJ | R | donor, ligand centre | RMSD permutation | none |
| FEXWUC | C | ligand centre | RMSD permutation | none |
| HOYBED | Z | donor, ligand centre | none | donor |
| SILROY | C | ligand centre | mirror tolerance | none |
| USAPOU | Z | ligand centre | none | donor |
| UYOVUA | R | donor | RMSD permutation | none |
| YOVNEC | Z | double bond | none | double bond |
| ZAQNUE | C | donor | mirror tolerance | none |
| ZAZREZ | Z | double bond, metal | metal exclusion | double bond |

The 7 parents that leave Z do so as follows: RMSD permutation resolves BABNAV, CAQLEP, FEKKAJ, FEXWUC and UYOVUA; mirror tolerance resolves SILROY and ZAQNUE. BABNAV, FEKKAJ and UYOVUA become R because ring share exceeds 0.5; CAQLEP, FEXWUC, SILROY and ZAQNUE become C. The six parents that remain Z have at least one mapped double-bond or donor-centre difference under Amendment 2.

## Parent decision table

`pool rows` gives seeds 11/12/13. `best RMSD` is the lowest symmetric heavy-atom RMSD to the relaxed reference; `lowest offset` is in kJ/mol. `shells` is the single-linkage count at 0.3 Å. Geometry values are the reference minimum CSM and builder-label CSM when defined. Donor IDs are zero-based atom indices. The Amendment 2 stereo column is populated for former Z parents.

| Parent | Class A2 | Class A1 | Seed 11/12/13 A2 | Geometry evidence | Donor evidence | Amendment 2 stereo details | Pool rows | Shells | Lowest offset | Best RMSD Å | Ring share | dedup_capped |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| BABNAV | R | Z | R/R/R | ref trigonal 0.373; builder trigonal 0.373; mismatch=False | builder `[1, 2, 5]`, ref06 `[1, 2, 5]` | old: ligand centre; removed: RMSD permutation; remains: none | 16/16/16 | 1 | 9.717 | 1.049 | 0.786 | False |
| BOCRAN | Z | Z | Z/Z/Z | ref trig_bipyr 0.290; builder trig_bipyr 0.290; mismatch=False | builder `[2, 3, 4, 5, 32]`, ref06 `[2, 3, 4, 5, 32]` | old: double bond,ligand centre; removed: mirror tolerance; remains: double bond | 16/16/16 | 1 | 15.289 | 3.867 | 0.867 | False |
| CAQLEP | C | Z | C/C/C | ref square_plan 0.221; builder square_plan 0.221; mismatch=False | builder `[1, 2, 3, 70]`, ref06 `[1, 2, 3, 70]` | old: ligand centre,metal; removed: metal exclusion,RMSD permutation; remains: none | 16/16/16 | 1 | 10.690 | 0.620 | 0.012 | False |
| CIFZEZ | Z | Z | Z/Z/Z | ref octahedral 0.572; builder octahedral 0.572; mismatch=False | builder `[1, 2, 3, 4, 5, 7]`, ref06 `[1, 2, 3, 4, 5, 7]` | old: double bond; removed: none; remains: double bond | 32/32/32 | 2 | 20.322 | 1.775 | 0.880 | True |
| CIXYUD | R | R | R/R/R | ref tetrahedral 0.086; builder tetrahedral 0.086; mismatch=False | builder `[1, 2, 3, 6]`, ref06 `[1, 2, 3, 6]` | — | 32/32/32 | 1 | 21.344 | 3.399 | 0.838 | True |
| DAMVAQ | T | T | T/T/T | ref trig_pris 28.949; builder t_shaped —; mismatch=False | builder `[1, 10, 15]`, ref06 `[1, 2, 8, 10, 11, 15]`; ref03 `[1, 2, 8, 10, 15]` (mismatch=True) | — | 16/16/16 | 4 | 168.837 | 1.084 | — | False |
| DOHTOM | R | R | R/R/R | ref octahedral 0.854; builder octahedral 0.854; mismatch=False | builder `[1, 2, 4, 15, 48, 51]`, ref06 `[1, 2, 4, 15, 48, 51]` | — | 32/32/32 | 1 | 67.077 | 0.545 | 0.962 | False |
| DUHHIX | R | R | R/R/R | ref octahedral 0.522; builder octahedral 0.522; mismatch=False | builder `[1, 2, 3, 4, 6, 32]`, ref06 `[1, 2, 3, 4, 6, 32]` | — | 64/64/64 | 6 | 84.516 | 0.960 | 0.821 | True |
| EFOXII | R | R | R/R/R | ref octahedral 0.002; builder octahedral 0.002; mismatch=False | builder `[3, 5, 8, 40, 42, 45]`, ref06 `[3, 5, 8, 40, 42, 45]` | — | 32/32/32 | 1 | 8.955 | 0.889 | 0.912 | False |
| EZIBUL | G | G | G/G/G | ref see_saw 0.683; builder square_plan 4.955; mismatch=True | builder `[1, 2, 3, 4]`, ref06 `[1, 2, 3, 4]` | — | 32/32/32 | 3 | 32.255 | 0.620 | 0.725 | False |
| FEKKAJ | R | Z | R/R/R | ref square_pyr 0.901; builder square_pyr 0.901; mismatch=False | builder `[1, 2, 3, 6, 53]`, ref06 `[1, 2, 3, 6, 53]` | old: donor,ligand centre; removed: RMSD permutation; remains: none | 144/144/144 | 7 | 47.574 | 0.674 | 0.902 | False |
| FETTAZ | F | F | F/F/F | ref octahedral 0.453; builder octahedral 0.453; mismatch=False | builder `[3, 4, 5, 23, 24, 25]`, ref06 `[3, 4, 5, 23, 24, 25]` | — | 80/80/80 | 1 | -38.413 | 0.687 | 0.149 | False |
| FEXWUC | C | Z | C/C/C | ref octahedral 0.626; builder octahedral 0.626; mismatch=False | builder `[3, 4, 10, 35, 36, 42]`, ref06 `[3, 4, 10, 35, 36, 42]` | old: ligand centre; removed: RMSD permutation; remains: none | 80/80/80 | 1 | 15.830 | 0.403 | 0.303 | False |
| FUTSOD | T | T | T/T/T | ref pent_bipyr 7.175; builder octahedral —; mismatch=False | builder `[1, 3, 5, 8, 25, 42]`, ref06 `[1, 2, 3, 5, 8, 25, 42]`; ref03 `[1, 3, 5, 8, 25, 42]` (mismatch=False) | — | 16/16/16 | 1 | 37.451 | 0.933 | 0.328 | False |
| GOTWET | T | T | T/T/T | undefined, CN 11: Si,Si,Si,N,N,N,O,O,O,C,C | builder `[4, 5, 6, 9, 34]`, ref06 `[1, 2, 3, 4, 5, 6, 7, 8, 9, 34, 74]`; ref03 `[4, 5, 6, 7, 9, 34]` (mismatch=True) | — | 64/64/64 | 152 | 18.774 | 2.227 | — | True |
| HOHZAG | R | R | R/R/R | ref square_pyr 0.142; builder square_pyr 0.142; mismatch=False | builder `[1, 2, 3, 4, 89]`, ref06 `[1, 2, 3, 4, 89]` | — | 19/19/16 | 2 | 16.866 | 0.345 | 0.984 | False |
| HOXJUZ | S | S | S/S/S | ref see_saw 1.672; builder square_plan 2.357; mismatch=False | builder `[1, 2, 3, 5]`, ref06 `[1, 2, 3, 5]` | — | 10/9/10 | 1 | 130.655 | 1.896 | — | False |
| HOYBED | Z | Z | Z/Z/Z | ref pent_bipyr 48.087; builder linear —; mismatch=False | builder `[1, 73, 74, 76, 78, 80, 82]`, ref06 `[1, 73, 74, 76, 78, 80, 82]` | old: donor,ligand centre; removed: none; remains: donor | 13/12/13 | 1 | 7.071 | 0.534 | 0.266 | False |
| LIPDEU | C | C | C/C/C | ref square_plan 0.796; builder square_plan 0.796; mismatch=False | builder `[1, 2, 6, 7]`, ref06 `[1, 2, 6, 7]` | — | 32/32/32 | 1 | 181.237 | 1.611 | 0.090 | False |
| LUQVOK | F | F | C/C/F | ref square_plan 0.045; builder square_plan 0.045; mismatch=False | builder `[0, 54, 57, 111]`, ref06 `[0, 54, 57, 111]` | — | 32/32/32 | 1 | 2.889 | 0.580 | 0.325 | False |
| MIDQEY | F | F | F/F/F | ref square_pyr 2.075; builder square_pyr 2.075; mismatch=False | builder `[1, 2, 3, 4, 7, 12, 27]`, ref06 `[1, 2, 3, 4, 7]` | — | 96/96/96 | 1 | -3.471 | 1.826 | 0.621 | False |
| NUCKUS | T | T | T/T/T | ref pent_bipyr 25.662; builder tetrahedral —; mismatch=False | builder `[1, 33, 67, 101]`, ref06 `[1, 33, 34, 67, 68, 101, 102]`; ref03 `[1, 33, 67, 101]` (mismatch=False) | — | 16/16/16 | 21 | 12.384 | 2.113 | — | True |
| NUWFAM | F | F | F/F/F | undefined, CN 14: C,C,C,C,C,C,C,C,C,C,C,C,C,F | builder `[1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`, ref06 `[1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 38]` | — | 32/32/32 | 33 | -14.595 | 0.927 | — | False |
| QAPFAS | C | C | C/C/C | ref linear 0.000; builder linear 0.000; mismatch=False | builder `[1, 47]`, ref06 `[1, 47]` | — | 1/1/1 | 1 | 6.575 | 1.932 | 0.284 | False |
| SILROY | C | Z | C/C/C | ref linear 0.000; builder linear 0.000; mismatch=False | builder `[1, 2]`, ref06 `[1, 2]` | old: ligand centre; removed: mirror tolerance; remains: none | 16/16/16 | 1 | 36.769 | 1.933 | 0.184 | True |
| SIVBUV | C | C | C/C/C | ref octahedral 0.168; builder octahedral 0.168; mismatch=False | builder `[1, 2, 3, 4, 5, 6]`, ref06 `[1, 2, 3, 4, 5, 6]` | — | 4/5/3 | 1 | 9.116 | 1.025 | 0.000 | False |
| USAPOU | Z | Z | Z/Z/Z | ref square_anti 7.194; builder octahedral —; mismatch=False | builder `[40, 42, 50, 52, 60, 61, 62, 63]`, ref06 `[40, 42, 50, 52, 60, 61, 62, 63]` | old: ligand centre; removed: none; remains: donor | 80/80/80 | 33 | 9.736 | 1.145 | 0.365 | False |
| UYOVUA | R | Z | R/R/R | ref square_plan 0.327; builder square_plan 0.327; mismatch=False | builder `[5, 6, 7, 8]`, ref06 `[5, 6, 7, 8]` | old: donor; removed: RMSD permutation; remains: none | 1/1/1 | 1 | 20.509 | 0.303 | 0.922 | False |
| XACMOE | S | S | S/S/S | ref octahedral 3.363; builder octahedral 3.363; mismatch=False | builder `[27, 28, 29, 30, 31, 32]`, ref06 `[27, 28, 29, 30, 31, 32]` | — | 23/18/19 | 12 | 10.464 | 0.633 | — | False |
| XEVDAE | T | T | T/T/T | undefined, CN 11: S,O,C,C,C,C,C,C,C,C,C | builder `[2, 7, 13, 14, 15, 22, 23, 24]`, ref06 `[1, 2, 7, 11, 12, 13, 14, 15, 22, 23, 24]`; ref03 `[2, 7, 13, 14, 15, 22, 23, 24]` (mismatch=False) | — | 16/16/16 | 39 | 6.161 | 0.683 | — | False |
| XOJHIN | F | F | F/C/C | ref trig_pris 9.559; builder square_plan —; mismatch=False | builder `[1, 4, 17, 18, 21, 22]`, ref06 `[1, 4, 17, 18, 21, 22]` | — | 32/32/32 | 2 | 4.897 | 0.377 | 0.019 | False |
| YOVNEC | Z | Z | Z/Z/Z | ref square_plan 0.007; builder square_plan 0.007; mismatch=False | builder `[1, 2, 3, 4]`, ref06 `[1, 2, 3, 4]` | old: double bond; removed: none; remains: double bond | 32/32/32 | 2 | 6.917 | 0.558 | 0.182 | True |
| ZAFPOP | R | R | R/R/R | ref linear 0.000; builder linear 0.000; mismatch=False | builder `[6, 73]`, ref06 `[6, 73]` | — | 32/32/32 | 1 | 7.259 | 0.755 | 0.848 | False |
| ZAQNUE | C | Z | C/C/C | ref square_plan 0.221; builder square_plan 0.221; mismatch=False | builder `[1, 2, 50, 64]`, ref06 `[1, 2, 50, 64]` | old: donor; removed: mirror tolerance; remains: none | 16/16/16 | 1 | 73.326 | 1.378 | 0.434 | False |
| ZAZREZ | Z | Z | Z/Z/Z | ref square_plan 0.001; builder square_plan 0.001; mismatch=False | builder `[1, 2, 3, 4]`, ref06 `[1, 2, 3, 4]` | old: double bond,metal; removed: metal exclusion; remains: double bond | 2/2/1 | 1 | 10.557 | 1.923 | 0.267 | False |

## Class counts and shell sensitivity

The all-seed Amendment 1 and Amendment 2 totals are shown side by side; seed and threshold rows use Amendment 2. `G*` is explicit and zero.

| Setting | F | T | G | G* | S | Z | R | C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amendment 1, relaxed shell ≤0.3 Å | 5 | 5 | 1 | 0 | 2 | 13 | 6 | 3 |
| Amendment 2, relaxed shell ≤0.3 Å | 5 | 5 | 1 | 0 | 2 | 6 | 9 | 7 |
| Seed 11, Amendment 2 | 4 | 5 | 1 | 0 | 2 | 6 | 9 | 8 |
| Seed 12, Amendment 2 | 3 | 5 | 1 | 0 | 2 | 6 | 9 | 9 |
| Seed 13, Amendment 2 | 4 | 5 | 1 | 0 | 2 | 6 | 9 | 8 |
| Relaxed shell ≤0.5 Å, Amendment 2 | 5 | 5 | 1 | 0 | 0 | 6 | 9 | 9 |
| Unrelaxed shell ≤0.3 Å, Amendment 2 | 5 | 5 | 1 | 0 | 0 | 6 | 9 | 9 |
| Unrelaxed shell ≤0.5 Å, Amendment 2 | 5 | 5 | 1 | 0 | 0 | 6 | 9 | 9 |

## Amendment 3 reanalysis (supersedes Amendment 2 classes)

The decision list was rerun locally from all 3,516 stored A4t relaxation rows. No relaxation, solve, or VM work was performed. A double bond is compared only when both structures have a definite stereo label; absent labels and `STEREOANY` on either side are skipped. The donor cause is assigned only when a centre that actually differs is a donor.

### Amendment 2 versus Amendment 3 class counts

| Rule set | F | T | G | G* | S | Z | R | C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amendment 2 | 5 | 5 | 1 | 0 | 2 | 6 | 9 | 7 |
| Amendment 3 | 5 | 5 | 1 | 0 | 2 | 4 | 9 | 9 |

YOVNEC and ZAZREZ leave Z because their respective double bonds are labelled `STEREOANY` on one side. Both now fall through to C. Remaining differences below use zero-based reference atom indices. “Metal-bound” means directly coordinated to the metal in the builder donor set. The selected row is the lowest-heavy-atom-RMSD row among relaxed rows with shell RMSD ≤0.3 Å.

### Remaining Z parents, atom-level causes

| Parent | Best row (seed, pool row) | Atom-level label differences | Amendment 3 cause keeping Z |
|---|---|---|---|
| BOCRAN | 12, 9 | N (idx 6, not metal-bound)=C (idx 32, metal-bound): E→Z. C (idx 46, not metal-bound): R→S, accepted as a global centre inversion. | Definite E/Z difference at N(idx 6)=C(idx 32); one endpoint is metal-bound. |
| CIFZEZ | 12, 1 | C (idx 7, metal-bound)=C (idx 16, not metal-bound): E→Z. | Definite E/Z difference at C(idx 7)=C(idx 16); one endpoint is metal-bound. |
| HOYBED | 11, 2 | Si (idx 1, metal-bound): R→S. C (idx 15) has a centre label on only one side and is skipped. | Differing metal-bound donor centre Si(idx 1). |
| USAPOU | 12, 28 | C (idx 10, not metal-bound): S→R; C (idx 28, not metal-bound): S→R. C (idx 16) has a centre label on only one side and is skipped. | Differing ligand centres C(idx 10) and C(idx 28); the donor cause label is not applicable. |

## Class F pool ranks

Every hit/equivalent row rank is listed by seed (1-based score order); empty lists mean no F row in that seed.

| Parent | Seed 11 ranks | Seed 12 ranks | Seed 13 ranks |
|---|---|---|---|
| FETTAZ | [9, 45, 49, 52, 56, 59, 66, 70] | [9, 44, 51, 52, 56, 57, 62, 66, 78] | [8, 30, 52, 70, 71, 73] |
| LUQVOK | [] | [] | [2] |
| MIDQEY | [3, 7, 10, 54] | [3, 5, 8, 14, 24, 43] | [2, 7, 11, 20, 25, 36, 51] |
| NUWFAM | [8, 13, 16, 30, 31] | [3, 8] | [3] |
| XOJHIN | [27] | [] | [] |

## Class × `dedup_capped` (Amendment 2)

| Class | `False` | `True` | Total |
|---|---:|---:|---:|
| F | 5 | 0 | 5 |
| T | 3 | 2 | 5 |
| G | 1 | 0 | 1 |
| G* | 0 | 0 | 0 |
| S | 2 | 0 | 2 |
| Z | 4 | 2 | 6 |
| R | 7 | 2 | 9 |
| C | 6 | 1 | 7 |

## Per-seed detail (Amendment 2)

| Parent | Seed | Class | Pool rows | OK | Capped | Failed | Hits | Equivalent | Shell matches ≤0.3 Å | Best RMSD Å | Lowest offset kJ/mol |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BABNAV | 11 | R | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 2.107 | 9.717 |
| BABNAV | 12 | R | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 1.582 | 15.221 |
| BABNAV | 13 | R | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 1.049 | 15.172 |
| BOCRAN | 11 | Z | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 4.014 | 15.289 |
| BOCRAN | 12 | Z | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 3.867 | 15.672 |
| BOCRAN | 13 | Z | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 4.062 | 16.653 |
| CAQLEP | 11 | C | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 0.883 | 12.414 |
| CAQLEP | 12 | C | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 0.620 | 10.690 |
| CAQLEP | 13 | C | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 0.774 | 17.320 |
| CIFZEZ | 11 | Z | 32 | 32 | 0 | 0 | 0 | 0 | 22 | 2.074 | 20.322 |
| CIFZEZ | 12 | Z | 32 | 32 | 0 | 0 | 0 | 0 | 24 | 1.775 | 25.221 |
| CIFZEZ | 13 | Z | 32 | 32 | 0 | 0 | 0 | 0 | 18 | 2.247 | 31.317 |
| CIXYUD | 11 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 3.399 | 21.344 |
| CIXYUD | 12 | R | 32 | 32 | 0 | 0 | 0 | 0 | 29 | 3.906 | 26.553 |
| CIXYUD | 13 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 4.156 | 22.038 |
| DAMVAQ | 11 | T | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 1.156 | 168.837 |
| DAMVAQ | 12 | T | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 1.084 | 176.985 |
| DAMVAQ | 13 | T | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 1.207 | 170.915 |
| DOHTOM | 11 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 0.583 | 83.326 |
| DOHTOM | 12 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 0.545 | 73.706 |
| DOHTOM | 13 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 0.585 | 67.077 |
| DUHHIX | 11 | R | 64 | 64 | 0 | 0 | 0 | 0 | 27 | 0.960 | 84.516 |
| DUHHIX | 12 | R | 64 | 64 | 0 | 0 | 0 | 0 | 20 | 1.024 | 89.450 |
| DUHHIX | 13 | R | 64 | 64 | 0 | 0 | 0 | 0 | 28 | 1.070 | 93.343 |
| EFOXII | 11 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 0.889 | 9.184 |
| EFOXII | 12 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 0.995 | 8.955 |
| EFOXII | 13 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 0.951 | 10.869 |
| EZIBUL | 11 | G | 32 | 32 | 0 | 0 | 0 | 0 | 25 | 0.620 | 41.686 |
| EZIBUL | 12 | G | 32 | 32 | 0 | 0 | 0 | 0 | 24 | 0.881 | 32.255 |
| EZIBUL | 13 | G | 32 | 32 | 0 | 0 | 0 | 0 | 24 | 0.817 | 34.289 |
| FEKKAJ | 11 | R | 144 | 144 | 0 | 0 | 0 | 0 | 60 | 0.787 | 65.952 |
| FEKKAJ | 12 | R | 144 | 144 | 0 | 0 | 0 | 0 | 56 | 0.674 | 67.528 |
| FEKKAJ | 13 | R | 144 | 144 | 0 | 0 | 0 | 0 | 50 | 0.807 | 47.574 |
| FETTAZ | 11 | F | 80 | 80 | 0 | 0 | 0 | 8 | 78 | 0.687 | -38.413 |
| FETTAZ | 12 | F | 80 | 80 | 0 | 0 | 0 | 9 | 79 | 0.698 | -14.847 |
| FETTAZ | 13 | F | 80 | 80 | 0 | 0 | 0 | 6 | 79 | 0.709 | -11.321 |
| FEXWUC | 11 | C | 80 | 80 | 0 | 0 | 0 | 0 | 80 | 0.761 | 20.844 |
| FEXWUC | 12 | C | 80 | 80 | 0 | 0 | 0 | 0 | 80 | 0.403 | 15.830 |
| FEXWUC | 13 | C | 80 | 80 | 0 | 0 | 0 | 0 | 80 | 0.661 | 21.001 |
| FUTSOD | 11 | T | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 0.933 | 37.451 |
| FUTSOD | 12 | T | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 1.133 | 42.528 |
| FUTSOD | 13 | T | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 1.314 | 43.998 |
| GOTWET | 11 | T | 64 | 64 | 0 | 0 | 0 | 0 | 0 | 2.227 | 30.185 |
| GOTWET | 12 | T | 64 | 64 | 0 | 0 | 0 | 0 | 0 | 2.311 | 31.946 |
| GOTWET | 13 | T | 64 | 64 | 0 | 0 | 0 | 0 | 0 | 2.247 | 18.774 |
| HOHZAG | 11 | R | 19 | 19 | 0 | 0 | 0 | 0 | 18 | 0.345 | 16.866 |
| HOHZAG | 12 | R | 19 | 19 | 0 | 0 | 0 | 0 | 17 | 0.371 | 20.837 |
| HOHZAG | 13 | R | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 0.357 | 20.173 |
| HOXJUZ | 11 | S | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 1.914 | 130.655 |
| HOXJUZ | 12 | S | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 1.908 | 151.924 |
| HOXJUZ | 13 | S | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 1.896 | 136.661 |
| HOYBED | 11 | Z | 13 | 13 | 0 | 0 | 0 | 0 | 13 | 0.534 | 7.071 |
| HOYBED | 12 | Z | 12 | 12 | 0 | 0 | 0 | 0 | 12 | 0.617 | 8.103 |
| HOYBED | 13 | Z | 13 | 13 | 0 | 0 | 0 | 0 | 13 | 0.551 | 7.718 |
| LIPDEU | 11 | C | 32 | 32 | 0 | 0 | 0 | 0 | 22 | 1.611 | 181.237 |
| LIPDEU | 12 | C | 32 | 32 | 0 | 0 | 0 | 0 | 25 | 1.669 | 183.520 |
| LIPDEU | 13 | C | 32 | 32 | 0 | 0 | 0 | 0 | 25 | 1.654 | 183.786 |
| LUQVOK | 11 | C | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 0.799 | 12.901 |
| LUQVOK | 12 | C | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 0.935 | 5.107 |
| LUQVOK | 13 | F | 32 | 32 | 0 | 0 | 0 | 1 | 32 | 0.580 | 2.889 |
| MIDQEY | 11 | F | 96 | 96 | 0 | 0 | 0 | 4 | 29 | 1.875 | -3.471 |
| MIDQEY | 12 | F | 96 | 96 | 0 | 0 | 0 | 6 | 27 | 1.826 | -1.569 |
| MIDQEY | 13 | F | 96 | 96 | 0 | 0 | 0 | 7 | 29 | 1.854 | -2.409 |
| NUCKUS | 11 | T | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 2.737 | 17.815 |
| NUCKUS | 12 | T | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 2.113 | 18.066 |
| NUCKUS | 13 | T | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 2.284 | 12.384 |
| NUWFAM | 11 | F | 32 | 32 | 0 | 0 | 0 | 5 | 0 | 0.927 | -14.595 |
| NUWFAM | 12 | F | 32 | 32 | 0 | 0 | 0 | 2 | 0 | 1.412 | -1.657 |
| NUWFAM | 13 | F | 32 | 32 | 0 | 0 | 0 | 1 | 0 | 1.018 | -1.119 |
| QAPFAS | 11 | C | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1.935 | 6.965 |
| QAPFAS | 12 | C | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1.934 | 6.693 |
| QAPFAS | 13 | C | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1.932 | 6.575 |
| SILROY | 11 | C | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 1.933 | 38.971 |
| SILROY | 12 | C | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 2.053 | 36.769 |
| SILROY | 13 | C | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 2.034 | 38.488 |
| SIVBUV | 11 | C | 4 | 4 | 0 | 0 | 0 | 0 | 4 | 1.075 | 9.116 |
| SIVBUV | 12 | C | 5 | 5 | 0 | 0 | 0 | 0 | 5 | 1.025 | 10.761 |
| SIVBUV | 13 | C | 3 | 3 | 0 | 0 | 0 | 0 | 3 | 1.040 | 10.728 |
| USAPOU | 11 | Z | 80 | 80 | 0 | 0 | 0 | 0 | 19 | 1.211 | 14.354 |
| USAPOU | 12 | Z | 80 | 80 | 0 | 0 | 0 | 0 | 20 | 1.145 | 12.720 |
| USAPOU | 13 | Z | 80 | 80 | 0 | 0 | 0 | 0 | 18 | 1.195 | 9.736 |
| UYOVUA | 11 | R | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0.303 | 20.510 |
| UYOVUA | 12 | R | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0.303 | 20.509 |
| UYOVUA | 13 | R | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0.303 | 20.509 |
| XACMOE | 11 | S | 23 | 23 | 0 | 0 | 0 | 0 | 0 | 0.652 | 18.063 |
| XACMOE | 12 | S | 18 | 18 | 0 | 0 | 0 | 0 | 0 | 0.633 | 19.004 |
| XACMOE | 13 | S | 19 | 19 | 0 | 0 | 0 | 0 | 0 | 0.897 | 10.464 |
| XEVDAE | 11 | T | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 0.683 | 6.161 |
| XEVDAE | 12 | T | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 0.695 | 18.483 |
| XEVDAE | 13 | T | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 0.750 | 8.915 |
| XOJHIN | 11 | F | 32 | 32 | 0 | 0 | 0 | 1 | 32 | 0.413 | 4.897 |
| XOJHIN | 12 | C | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 0.377 | 14.077 |
| XOJHIN | 13 | C | 32 | 32 | 0 | 0 | 0 | 0 | 31 | 0.439 | 19.820 |
| YOVNEC | 11 | Z | 32 | 32 | 0 | 0 | 0 | 0 | 30 | 1.217 | 9.967 |
| YOVNEC | 12 | Z | 32 | 32 | 0 | 0 | 0 | 0 | 30 | 0.558 | 10.014 |
| YOVNEC | 13 | Z | 32 | 32 | 0 | 0 | 0 | 0 | 30 | 1.017 | 6.917 |
| ZAFPOP | 11 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 1.102 | 9.528 |
| ZAFPOP | 12 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 0.755 | 9.450 |
| ZAFPOP | 13 | R | 32 | 32 | 0 | 0 | 0 | 0 | 32 | 1.051 | 7.259 |
| ZAQNUE | 11 | C | 16 | 16 | 0 | 0 | 0 | 0 | 15 | 1.378 | 73.326 |
| ZAQNUE | 12 | C | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 1.490 | 84.189 |
| ZAQNUE | 13 | C | 16 | 16 | 0 | 0 | 0 | 0 | 16 | 1.409 | 86.924 |
| ZAZREZ | 11 | Z | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 1.923 | 10.707 |
| ZAZREZ | 12 | Z | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 2.023 | 10.557 |
| ZAZREZ | 13 | Z | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 2.112 | 11.012 |

## Deviations and failures

No pool row failed, capped, or lacked a relaxation result. The first local analysis attempt spent about 26 CPU minutes in scalar permutation loops and was stopped; the exact batched Kabsch implementation was used thereafter. The first complete amended analysis initially hit an exit-137 termination at a 12-donor permutation setup; lazy permutation generation capped the high-symmetry enumeration at 40,320 without materializing the full factorial, and the rerun completed all 35 parents and 3,516 rows. The shell-cluster computation completed for every parent. All data and class counts above use the completed rerun.
