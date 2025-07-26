
# For Fig.1 and Fig.2 in Section 4
Simulation Results for Normal Distribution
for n in $(seq 200 100 1000);
do
    python splitmodelx_test.py --D_types "0,1,2" --n ${n} --p 100 --k 20 --A 1 --c 0.3 --lambdas="-4.0:-1.8:0.2" --nus="2.0:4.2:0.2" --model_type logistic
done


# For Fig3 in Section 4 (Running time of Fig4 is also included)
# Simulation Results for Pariwise Comparison
for p in $(seq 10 5 30);
do
    # python simu_exp/splitmodelx_pairwise_test.py --D_types "0" --lambdas="-3.0:1.0:1.0" --nus="0.0:1.2:0.2" --n 1000 --p ${p} --A 5 --k 0.5 --c 0.3 --data_type "normal" --con_type "pairwise_seqM" --model_type "logistic" --Z_types db
    python splitmodelx_pairwise_test.py --D_types "0" --lambdas="-6.0:1.0:1.0" --nus="3.0:6.0:1.0" --n 1000 --p ${p} --A 5 --k 0.5 --c 0.3 --data_type "resample" --con_type "pairwise_fixaugM" --model_type "logistic" --Z_types db
done

# For Fig5 in Section 4
# Simulation Results for Pariwise Comparison
for n in $(seq 100 100 1000);
do
    python simu_exp/splitmodelx_pairwise_test.py --D_types "0" --n ${n} --p 10 --A 3 --k 0.1 --c 0.3 --lambdas="-6.0:0.0:1.0" --nus="2.0:4.0:0.5" --data_type "resample" --con_type "pairwise_fixaugM" --model_type "logistic"
    python simu_exp/splitmodelx_pairwise_test.py --D_types "0" --n ${n} --p 10 --A 3 --k 0.3 --c 0.3 --lambdas="-6.0:0.0:1.0" --nus="2.0:4.0:0.5" --data_type "resample" --con_type "pairwise_fixaugM" --model_type "logistic"
    python simu_exp/splitmodelx_pairwise_test.py --D_types "0" --n ${n} --p 10 --A 3 --k 0.5 --c 0.3 --lambdas="-6.0:0.0:1.0" --nus="2.0:4.0:0.5" --data_type "resample" --con_type "pairwise_fixaugM" --model_type "logistic"
done
