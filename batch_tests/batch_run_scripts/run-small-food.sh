source ../set_env.sh
c=1
while IFS='	' read -r data model sub name count dcs max_dim location
do
    echo $location
    echo $name
    for feat in init lang initsim occurattr constraint initattr freq
    do
        ./../create_db.sh food_small_v${c}
        python hc.py -msg validation -notes v${c} -dataname food_small -dcpath $location -dc $name -k 0.7 -w 0.01 -omit occur $feat  --wlog &> ../log/small_food/omit-${feat}.log
        echo $c
        c=$((c+1))
    done
    python send_email.py small_food
done < ../batch_run_dcs/dc-food-small.txt
