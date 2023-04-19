for ((i=1; i<=1600; i++))
do	
	j=$(($i + 1600))
	mv ${i}.png train/${i}.png
done

for ((i=1601; i<=2000; i++))
do
	j=$(($i - 1600))
	mv ${i}.png test/${j}.png
done
