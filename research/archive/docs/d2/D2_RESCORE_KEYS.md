# D2 rescore: concrete lookup keys

Captured on Perlmutter using committed `6fe90b5`, the same DBs and pools, with one logged xTB retry at 1000 K. All six rescore aggregates exactly matched the prior reference. These are keys with a changed resolved fit record (including metadata); presence does not prove that a given key contributed a particular amount to the score. Values and all captured changed keys are in main-checkout `research/data/d2/db_d2/rescore_changed_keys.json`.

## AJIXUN

| Quantity | Requested key | Production resolved key | D2 resolved key |
|---|---|---|---|
| Q5 | `["C_sp2","C_ar","C_sp2","C_ar","free_o0"]` | `["C_sp2","C_ar","C_sp2","C_ar","free_o0"]` | `["C_sp2","C_ar","C_sp2","C_ar","free_o0"]` |
| Q5 | `["P_sp3","C_ar","C_sp3","C_ar","free_o0"]` | `["P_sp3","C_ar","C_sp3","C_ar","free_o0"]` | `["P_sp3","C_ar","C_sp3","C_ar","free_o0"]` |
| Q5 | `["P_sp3","C_sp3","C_ar","H","free"]` | `["P_sp3","C_sp3","C_ar","H","free"]` | `["P_sp3","C_sp3","C_ar","H","free"]` |

## AFOWIB

| Quantity | Requested key | Production resolved key | D2 resolved key |
|---|---|---|---|
| Q5 | `["N_ar","C_sp3","C_ar","H","free"]` | `["N_ar","C_sp3","C_ar","H","free"]` | `["N_ar","C_sp3","C_ar","H","free"]` |
| Q5 | `["N_sp2","N_sp3","C_ar","C_ar","chelate"]` | `["N_sp2","N_sp3","C_ar","C_ar","chelate"]` | `["N_sp2","N_sp3","C_ar","C_ar","chelate"]` |
| Q5 | `["N_sp3","C_ar","N_sp2","C_ar","free_o0"]` | `["N_sp3","C_ar","N_sp2","C_ar","free_o0"]` | `["N_sp3","C_ar","N_sp2","C_ar","free_o0"]` |

## ADUJOB

| Quantity | Requested key | Production resolved key | D2 resolved key |
|---|---|---|---|
| Q1 | `["Ag",1,1,"C_allyl","square_pyr","C_allyl"]` | `["Ag",1,1,"C_allyl","square_pyr","C_allyl"]` | `["Ag",1,1,"C_allyl","square_pyr","C_allyl"]` |
| Q1 | `["Ag",1,1,"C_allyl","square_pyr","N_cyanide"]` | `["Ag",1,1,"C_allyl","square_pyr"]` | `["Ag",1,1,"C_allyl","square_pyr"]` |
| Q1 | `["Ag",1,1,"C_allyl","square_pyr","N_imine"]` | `["Ag",1,1,"C_allyl","square_pyr"]` | `["Ag",1,1,"C_allyl","square_pyr"]` |
| Q1 | `["Ag",1,1,"C_allyl","square_pyr","none"]` | `["Ag",1,1,"C_allyl","square_pyr"]` | `["Ag",1,1,"C_allyl","square_pyr"]` |
| Q1 | `["Ag",1,1,"N_cyanide","square_pyr","C_allyl"]` | `["Ag",1,1,"N_cyanide","square_pyr"]` | `["Ag",1,1,"N_cyanide","square_pyr"]` |
| Q1 | `["Ag",1,1,"N_cyanide","square_pyr","N_imine"]` | `["Ag",1,1,"N_cyanide","square_pyr"]` | `["Ag",1,1,"N_cyanide","square_pyr"]` |
| Q1 | `["Ag",1,1,"N_cyanide","square_pyr","none"]` | `["Ag",1,1,"N_cyanide","square_pyr"]` | `["Ag",1,1,"N_cyanide","square_pyr"]` |
| Q1 | `["Ag",1,1,"N_imine","square_pyr","C_allyl"]` | `["Ag",1,1,"N_imine","square_pyr"]` | `["Ag",1,1,"N_imine","square_pyr"]` |
| Q1 | `["Ag",1,1,"N_imine","square_pyr","N_cyanide"]` | `["Ag",1,1,"N_imine","square_pyr"]` | `["Ag",1,1,"N_imine","square_pyr"]` |
| Q1 | `["Ag",1,1,"N_imine","square_pyr","none"]` | `["Ag",1,1,"N_imine","square_pyr"]` | `["Ag",1,1,"N_imine","square_pyr"]` |

## AGAPEG

| Quantity | Requested key | Production resolved key | D2 resolved key |
|---|---|---|---|
| Q1 | `["Pd",1,2,"C_haptic","square_pyr","N_amide"]` | `["Pd",1,2,"C_haptic","square_pyr"]` | `["Pd",1,2,"C_haptic","square_pyr"]` |
| Q1 | `["Pd",1,2,"C_haptic","square_pyr","N_amine"]` | `["Pd",1,2,"C_haptic","square_pyr","N_amine"]` | `["Pd",1,2,"C_haptic","square_pyr"]` |
| Q1 | `["Pd",1,2,"C_haptic","square_pyr","N_pyridine"]` | `["Pd",1,2,"C_haptic","square_pyr","N_pyridine"]` | `["Pd",1,2,"C_haptic","square_pyr"]` |
| Q1 | `["Pd",1,2,"C_haptic","square_pyr","none"]` | `["Pd",1,2,"C_haptic","square_pyr"]` | `["Pd",1,2,"C_haptic","square_pyr"]` |
| Q1 | `["Pd",1,2,"N_amide","square_pyr","C_haptic"]` | `["Pd",1,2,"N_amide","square_pyr"]` | `["Pd",1,2,"N_amide","square_pyr"]` |
| Q1 | `["Pd",1,2,"N_amide","square_pyr","O_carboxylate"]` | `["Pd",1,2,"N_amide","square_pyr","O_carboxylate"]` | `["Pd",1,2,"N_amide","square_pyr"]` |
| Q1 | `["Pd",1,2,"N_amide","square_pyr","none"]` | `["Pd",1,2,"N_amide","square_pyr"]` | `["Pd",1,2,"N_amide","square_pyr"]` |
| Q1 | `["Pd",1,2,"N_amine","square_pyr","C_haptic"]` | `["Pd",1,2,"N_amine","square_pyr","C_haptic"]` | `["Pd",1,2,"N_amine","square_pyr"]` |
| Q1 | `["Pd",1,2,"N_amine","square_pyr","N_pyridine"]` | `["Pd",1,2,"N_amine","square_pyr","N_pyridine"]` | `["Pd",1,2,"N_amine","square_pyr","N_pyridine"]` |
| Q1 | `["Pd",1,2,"N_amine","square_pyr","O_carboxylate"]` | `["Pd",1,2,"N_amine","square_pyr","O_carboxylate"]` | `["Pd",1,2,"N_amine","square_pyr"]` |
| Q1 | `["Pd",1,2,"N_amine","square_pyr","none"]` | `["Pd",1,2,"N_amine","square_pyr"]` | `["Pd",1,2,"N_amine","square_pyr"]` |
| Q1 | `["Pd",1,2,"N_pyridine","square_pyr","C_haptic"]` | `["Pd",1,2,"N_pyridine","square_pyr","C_haptic"]` | `["Pd",1,2,"N_pyridine","square_pyr"]` |
| Q1 | `["Pd",1,2,"N_pyridine","square_pyr","N_amine"]` | `["Pd",1,2,"N_pyridine","square_pyr","N_amine"]` | `["Pd",1,2,"N_pyridine","square_pyr","N_amine"]` |
| Q1 | `["Pd",1,2,"N_pyridine","square_pyr","O_carboxylate"]` | `["Pd",1,2,"N_pyridine","square_pyr","O_carboxylate"]` | `["Pd",1,2,"N_pyridine","square_pyr"]` |
| Q1 | `["Pd",1,2,"N_pyridine","square_pyr","none"]` | `["Pd",1,2,"N_pyridine","square_pyr"]` | `["Pd",1,2,"N_pyridine","square_pyr"]` |
| Q1 | `["Pd",1,2,"O_carboxylate","square_pyr","N_amide"]` | `["Pd",1,2,"O_carboxylate","square_pyr","N_amide"]` | `["Pd",1,2,"O_carboxylate","square_pyr"]` |
| Q1 | `["Pd",1,2,"O_carboxylate","square_pyr","N_amine"]` | `["Pd",1,2,"O_carboxylate","square_pyr","N_amine"]` | `["Pd",1,2,"O_carboxylate","square_pyr"]` |
| Q1 | `["Pd",1,2,"O_carboxylate","square_pyr","N_pyridine"]` | `["Pd",1,2,"O_carboxylate","square_pyr","N_pyridine"]` | `["Pd",1,2,"O_carboxylate","square_pyr"]` |
| Q1 | `["Pd",1,2,"O_carboxylate","square_pyr","none"]` | `["Pd",1,2,"O_carboxylate","square_pyr"]` | `["Pd",1,2,"O_carboxylate","square_pyr"]` |
| Q2 | `["square_pyr","c180","chelate_long","Pd","N_amine","N_pyridine"]` | `["square_pyr","c180","chelate_long","Pd","N_amine","N_pyridine"]` | `["square_pyr","c180","chelate_long","Pd","N_donor","N_donor"]` |
| Q2 | `["square_pyr","c180","different_ligand","Pd","C_haptic","N_amide"]` | `["square_pyr","c180","different_ligand","Pd","C_haptic_any","N_sp3_any"]` | `["square_pyr","c180","different_ligand","Pd","C_haptic_any","N_sp3_any"]` |
| Q2 | `["square_pyr","c180","different_ligand","Pd","C_haptic","N_amine"]` | `["square_pyr","c180","different_ligand","Pd","C_haptic","N_amine"]` | `["square_pyr","c180","different_ligand","Pd","C_haptic_any","N_sp3_any"]` |
| Q2 | `["square_pyr","c180","different_ligand","Pd","C_haptic","N_pyridine"]` | `["square_pyr","c180","different_ligand","Pd","C_haptic","N_pyridine"]` | `["square_pyr","c180","different_ligand","Pd","C_haptic","N_pyridine"]` |
| Q2 | `["square_pyr","c180","different_ligand","Pd","N_amide","O_carboxylate"]` | `["square_pyr","c180","different_ligand","Pd","N_amide","O_carboxylate"]` | `["square_pyr","c180","different_ligand","Pd","N_sp3_any","O_anionic_any"]` |
| Q2 | `["square_pyr","c180","different_ligand","Pd","N_amine","O_carboxylate"]` | `["square_pyr","c180","different_ligand","Pd","N_amine","O_carboxylate"]` | `["square_pyr","c180","different_ligand","Pd","N_sp3_any","O_anionic_any"]` |
| Q2 | `["square_pyr","c180","different_ligand","Pd","N_pyridine","O_carboxylate"]` | `["square_pyr","c180","different_ligand","Pd","N_pyridine","O_carboxylate"]` | `["square_pyr","c180","different_ligand","Pd","N_aromatic_any","O_anionic_any"]` |
| Q2 | `["square_pyr","c90","chelate_long","Pd","N_amine","N_pyridine"]` | `["square_pyr","c90","chelate_long","Pd","N_donor","N_donor"]` | `["square_pyr","c90","chelate_long","Pd","N_donor","N_donor"]` |
| Q2 | `["square_pyr","c90","chelate_short","Pd","C_haptic","O_carboxylate"]` | `["square_pyr","c90","chelate_short","Pd","C_haptic","O_carboxylate"]` | `["square_pyr","c90","chelate_short","Pd","C_haptic_any","O_donor"]` |
| Q2 | `["square_pyr","c90","chelate_short","Pd","N_amide","N_amine"]` | `["square_pyr","c90","chelate_short","Pd","N_amide","N_amine"]` | `["square_pyr","c90","chelate_short","Pd","N_sp3_any","N_sp3_any"]` |
| Q2 | `["square_pyr","c90","chelate_short","Pd","N_amide","N_pyridine"]` | `["square_pyr","c90","chelate_short","Pd","N_amide","N_pyridine"]` | `["square_pyr","c90","chelate_short","Pd","N_amide","N_pyridine"]` |
| Q2 | `["square_pyr","c90","different_ligand","Pd","C_haptic","N_amide"]` | `["square_pyr","c90","different_ligand","Pd","C_haptic","N_amide"]` | `["square_pyr","c90","different_ligand","Pd","C_haptic_any","N_sp3_any"]` |
| Q2 | `["square_pyr","c90","different_ligand","Pd","C_haptic","N_amine"]` | `["square_pyr","c90","different_ligand","Pd","C_haptic","N_amine"]` | `["square_pyr","c90","different_ligand","Pd","C_haptic_any","N_sp3_any"]` |
| Q2 | `["square_pyr","c90","different_ligand","Pd","C_haptic","N_pyridine"]` | `["square_pyr","c90","different_ligand","Pd","C_haptic","N_pyridine"]` | `["square_pyr","c90","different_ligand","Pd","C_haptic","N_pyridine"]` |
| Q2 | `["square_pyr","c90","different_ligand","Pd","N_amide","O_carboxylate"]` | `["square_pyr","c90","different_ligand","Pd","N_amide","O_carboxylate"]` | `["square_pyr","c90","different_ligand","Pd","N_sp3_any","O_anionic_any"]` |
| Q2 | `["square_pyr","c90","different_ligand","Pd","N_amine","O_carboxylate"]` | `["square_pyr","c90","different_ligand","Pd","N_amine","O_carboxylate"]` | `["square_pyr","c90","different_ligand","Pd","N_sp3_any","O_anionic_any"]` |
| Q2 | `["square_pyr","c90","different_ligand","Pd","N_pyridine","O_carboxylate"]` | `["square_pyr","c90","different_ligand","Pd","N_pyridine","O_carboxylate"]` | `["square_pyr","c90","different_ligand","Pd","N_pyridine","O_carboxylate"]` |
| Q3 | `["N_amide","Pd","square_pyr",1]` | `["N_amide",1]` | `["N_amide","Pd",1]` |
| Q3 | `["N_amine","Pd","square_pyr",1]` | `["N_amine","Pd",1]` | `["N_amine","Pd",1]` |
| Q3 | `["N_pyridine","Pd","square_pyr",1]` | `["N_pyridine","Pd",1]` | `["N_pyridine","Pd",1]` |
| Q3 | `["O_carboxylate","Pd","square_pyr",0]` | `["O_carboxylate","Pd",0]` | `["O_carboxylate","Pd",0]` |

## AFAVUB

| Quantity | Requested key | Production resolved key | D2 resolved key |
|---|---|---|---|
| Q1 | `["Cd",1,2,"Hal_I","trig_pris","N_imine"]` | `["Cd",1,2,"Hal_I","trig_pris"]` | `["Cd",1,2,"Hal_I","trig_pris"]` |
| Q1 | `["Cd",1,2,"Hal_I","trig_pris","N_pyridine"]` | `["Cd",1,2,"Hal_I","trig_pris","N_pyridine"]` | `["Cd",1,2,"Hal_I","trig_pris","N_pyridine"]` |
| Q1 | `["Cd",1,2,"Hal_I","trig_pris","none"]` | `["Cd",1,2,"Hal_I","trig_pris"]` | `["Cd",1,2,"Hal_I","trig_pris"]` |
| Q1 | `["Cd",1,2,"N_imine","trig_pris","Hal_I"]` | `["Cd",1,2,"N_imine","trig_pris"]` | `["Cd",1,2,"N_imine","trig_pris"]` |
| Q1 | `["Cd",1,2,"N_imine","trig_pris","none"]` | `["Cd",1,2,"N_imine","trig_pris"]` | `["Cd",1,2,"N_imine","trig_pris"]` |
| Q1 | `["Cd",1,2,"N_pyridine","trig_pris","Hal_I"]` | `["Cd",1,2,"N_pyridine","trig_pris","Hal_I"]` | `["Cd",1,2,"N_pyridine","trig_pris","Hal_I"]` |
| Q1 | `["Cd",1,2,"N_pyridine","trig_pris","none"]` | `["Cd",1,2,"N_pyridine","trig_pris"]` | `["Cd",1,2,"N_pyridine","trig_pris"]` |
| Q2 | `["trig_pris","c139","chelate_long","Cd","N_imine","N_imine"]` | `["trig_pris","c139","chelate_long","Cd","N_imine","N_imine"]` | `["trig_pris","c139","chelate_long","Cd","N_imine","N_imine"]` |
| Q2 | `["trig_pris","c139","chelate_long","Cd","N_imine","N_pyridine"]` | `["trig_pris","c139","chelate_long","Cd","N_imine","N_pyridine"]` | `["trig_pris","c139","chelate_long","Cd","N_imine","N_pyridine"]` |
| Q2 | `["trig_pris","c139","chelate_short","Cd","N_imine","N_pyridine"]` | `["trig_pris","c139","chelate_short","Cd","N_imine","N_pyridine"]` | `["trig_pris","c139","chelate_short","period5_3d","N_donor","N_donor"]` |
| Q2 | `["trig_pris","c139","different_ligand","Cd","Hal_I","Hal_I"]` | `["trig_pris","c139","different_ligand","Cd","Hal_I","Hal_I"]` | `["trig_pris","c139","different_ligand","Cd","Hal_I","Hal_I"]` |
| Q2 | `["trig_pris","c139","different_ligand","Cd","Hal_I","N_imine"]` | `["trig_pris","c139","different_ligand","Cd","Hal_I","N_imine"]` | `["trig_pris","c139","different_ligand","Cd","Hal_I","N_imine"]` |
| Q2 | `["trig_pris","c139","different_ligand","Cd","Hal_I","N_pyridine"]` | `["trig_pris","c139","different_ligand","Cd","Hal_I","N_pyridine"]` | `["trig_pris","c139","different_ligand","Cd","Hal_I","N_pyridine"]` |
| Q2 | `["trig_pris","c139","same_remote","Cd","N_pyridine","N_pyridine"]` | `["trig_pris","c139","same_remote","Cd","N_pyridine","N_pyridine"]` | `["trig_pris","c139","same_remote","Cd","N_pyridine","N_pyridine"]` |
| Q2 | `["trig_pris","c76","chelate_long","Cd","N_imine","N_imine"]` | `["trig_pris","c76","chelate_long","Cd","N_sp3_any","N_sp3_any"]` | `["trig_pris","c76","chelate_long","Cd","N_sp3_any","N_sp3_any"]` |
| Q2 | `["trig_pris","c76","chelate_long","Cd","N_imine","N_pyridine"]` | `["trig_pris","c76","chelate_long","Cd","N_imine","N_pyridine"]` | `["trig_pris","c76","chelate_long","Cd","N_imine","N_pyridine"]` |
| Q2 | `["trig_pris","c76","chelate_short","Cd","N_imine","N_pyridine"]` | `["trig_pris","c76","chelate_short","Cd","N_imine","N_pyridine"]` | `["trig_pris","c76","chelate_short","Cd","N_imine","N_pyridine"]` |
| Q2 | `["trig_pris","c76","different_ligand","Cd","Hal_I","Hal_I"]` | `["trig_pris","c76","different_ligand","Cd","Hal_donor","Hal_donor"]` | `["trig_pris","c76","different_ligand","Cd","Hal_donor","Hal_donor"]` |
| Q2 | `["trig_pris","c76","different_ligand","Cd","Hal_I","N_imine"]` | `["trig_pris","c76","different_ligand","Cd","Hal_I","N_imine"]` | `["trig_pris","c76","different_ligand","Cd","Hal_I","N_imine"]` |
| Q2 | `["trig_pris","c76","different_ligand","Cd","Hal_I","N_pyridine"]` | `["trig_pris","c76","different_ligand","Cd","Hal_I","N_pyridine"]` | `["trig_pris","c76","different_ligand","Cd","Hal_I","N_pyridine"]` |
| Q2 | `["trig_pris","c76","same_remote","Cd","N_pyridine","N_pyridine"]` | `["trig_pris","c76","same_remote","Cd","N_pyridine","N_pyridine"]` | `["trig_pris","c76","same_remote","Cd","N_pyridine","N_pyridine"]` |
| Q2 | `["trig_pris","c90","chelate_long","Cd","N_imine","N_imine"]` | `["trig_pris","c90","chelate_long","Cd","N_imine","N_imine"]` | `["trig_pris","c90","chelate_long","Cd","N_imine","N_imine"]` |
| Q2 | `["trig_pris","c90","chelate_long","Cd","N_imine","N_pyridine"]` | `["trig_pris","c90","chelate_long","Cd","N_donor","N_donor"]` | `["trig_pris","c90","chelate_long","Cd","N_donor","N_donor"]` |
| Q2 | `["trig_pris","c90","chelate_short","Cd","N_imine","N_pyridine"]` | `["trig_pris","c90","chelate_short","Cd","N_imine","N_pyridine"]` | `["trig_pris","c90","chelate_short","Cd","N_imine","N_pyridine"]` |
| Q2 | `["trig_pris","c90","different_ligand","Cd","Hal_I","Hal_I"]` | `["trig_pris","c90","different_ligand","period5_3d","Hal_donor","Hal_donor"]` | `["trig_pris","c90","different_ligand","period5_3d","Hal_donor","Hal_donor"]` |
| Q2 | `["trig_pris","c90","different_ligand","Cd","Hal_I","N_imine"]` | `["trig_pris","c90","different_ligand","Cd","Hal_donor","N_sp3_any"]` | `["trig_pris","c90","different_ligand","Cd","Hal_donor","N_sp3_any"]` |
| Q2 | `["trig_pris","c90","different_ligand","Cd","Hal_I","N_pyridine"]` | `["trig_pris","c90","different_ligand","Cd","Hal_I","N_pyridine"]` | `["trig_pris","c90","different_ligand","Cd","Hal_I","N_pyridine"]` |
| Q2 | `["trig_pris","c90","same_remote","Cd","N_pyridine","N_pyridine"]` | `["trig_pris","c90","same_remote","Cd","N_pyridine","N_pyridine"]` | `["trig_pris","c90","same_remote","period5_3d","N_donor","N_donor"]` |

## ADINOR

| Quantity | Requested key | Production resolved key | D2 resolved key |
|---|---|---|---|
| Q1 | `["W",1,2,"C_carbonyl","tetrahedral","none"]` | `["W",1,2,"C_carbonyl","tetrahedral"]` | `["W",1,2,"C_carbonyl","tetrahedral"]` |
| Q1 | `["W",1,2,"C_eta5","tetrahedral","N_nitrosyl"]` | `["W",1,2,"C_eta5","tetrahedral"]` | `["W",1,2,"C_eta5","tetrahedral"]` |
| Q1 | `["W",1,2,"C_eta5","tetrahedral","none"]` | `["W",1,2,"C_eta5","tetrahedral"]` | `["W",1,2,"C_eta5","tetrahedral"]` |
| Q1 | `["W",1,2,"C_haptic","tetrahedral","none"]` | `["W",1,2,"C_haptic","tetrahedral"]` | `["W",1,2,"C_haptic","tetrahedral"]` |
| Q1 | `["W",1,2,"N_nitrosyl","tetrahedral","C_eta5"]` | `["W",1,2,"N_nitrosyl","tetrahedral"]` | `["W",1,2,"N_nitrosyl","tetrahedral"]` |
| Q1 | `["W",1,2,"N_nitrosyl","tetrahedral","none"]` | `["W",1,2,"N_nitrosyl","tetrahedral"]` | `["W",1,2,"N_nitrosyl","tetrahedral"]` |
| Q2 | `["tetrahedral","c109","different_ligand","W","C_carbonyl","C_eta5"]` | `["tetrahedral","c109","different_ligand","W","C_carbonyl","C_eta5"]` | `["tetrahedral","c109","different_ligand","W","C_carbonyl","C_eta5"]` |
| Q2 | `["tetrahedral","c109","different_ligand","W","C_carbonyl","C_haptic"]` | `["tetrahedral","c109","different_ligand","W","C_carbonyl","C_haptic"]` | `["tetrahedral","c109","different_ligand","W","C_carbonyl","C_haptic"]` |
| Q2 | `["tetrahedral","c109","different_ligand","W","C_carbonyl","N_nitrosyl"]` | `["tetrahedral","c109","different_ligand","W","C_carbonyl","N_nitrosyl"]` | `["tetrahedral","c109","different_ligand","W","C_carbonyl","N_nitrosyl"]` |
| Q2 | `["tetrahedral","c109","different_ligand","W","C_eta5","C_haptic"]` | `["tetrahedral","c109","different_ligand","W","C_eta5","C_haptic"]` | `["tetrahedral","c109","different_ligand","W","C_eta5","C_haptic"]` |
| Q2 | `["tetrahedral","c109","different_ligand","W","C_eta5","N_nitrosyl"]` | `["tetrahedral","c109","different_ligand","W","C_eta5","N_nitrosyl"]` | `["tetrahedral","c109","different_ligand","W","C_eta5","N_nitrosyl"]` |
| Q2 | `["tetrahedral","c109","different_ligand","W","C_haptic","N_nitrosyl"]` | `["tetrahedral","c109","different_ligand","W","C_haptic","N_nitrosyl"]` | `["tetrahedral","c109","different_ligand","W","C_haptic","N_nitrosyl"]` |
| Q3 | `["C_carbonyl","W","tetrahedral",0]` | `["C_carbonyl","W","tetrahedral",0]` | `["C_carbonyl","W","tetrahedral",0]` |
| Q3 | `["C_eta5","W","tetrahedral",0]` | `["C_eta5","W","tetrahedral",0]` | `["C_eta5","W","tetrahedral",0]` |
| Q3 | `["N_nitrosyl","W","tetrahedral",0]` | `["N_nitrosyl","W","tetrahedral",0]` | `["N_nitrosyl","W","tetrahedral",0]` |
