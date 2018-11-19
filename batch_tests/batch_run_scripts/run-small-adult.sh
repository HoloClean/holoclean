source ../set_env.sh
c=1
while IFS='	' read -r data model sub name count dcs max_dim location
do

    echo $location
    echo $name
    for feat in init lang initsim occurattr constraint initattr freq
    do
        ./../create_db.sh small_adult_1_s${c}
        python hc.py -notes s${c} -dataname small_adult_1 -dcpath $location -dc $name -k 0.1 -w 0.01 -omit $feat occur --wlog &> ../log/small_adult/omit-${feat}.log
        echo $c
        c=$((c+1))
    done
    python send_email.py hc2
    break
done < ../batch_run_dcs/dc-adult-joint.txt