source ../set_env.sh
c=1
while IFS='	' read -r data model sub name count dcs max_dim location
do

    echo $location
    echo $name
    for feat in init lang initsim occurattr constraint initattr freq
    do
        ./../create_db.sh small_census_01_v${c}
        python hc.py -notes v${c} -dataname small_census_01 -dcpath $location -dc $name -k 0.1 -w 0.01 -omit $feat occur --wlog &> ../log/small_census/omit-${feat}.log
        echo $c
        c=$((c+1))
    done
    python send_email.py hc2
    break
done < ../batch_run_dcs/dc-census-001.txt
