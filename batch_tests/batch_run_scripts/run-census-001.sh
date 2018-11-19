source ../../set_env.sh
c=3
while IFS='	' read -r data model sub name count dcs max_dim location
do
    ./../../create_db_ubuntu.sh ${data}_${c}
    echo $location
    echo $name
    python hc.py -msg GL -notes $c -dataname $data -dcpath $location -dc $name -k 0.1 -w 0.01 -omit init lang occur --wlog &> ../log/census_001/runtime-${c}.log
    c=$((c+1))
    python send_email.py census
    
done < ../batch_scripts/dc-census-001.txt
