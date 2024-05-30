a1=0.283794109  # asin(7/25)
a2=0.532504098  # asin(33/65)
a3=0.781214087  # asin(119/169)
a4=1.03829223  # asin(56/65)
a5=1.28700222  # asin(24/25)
a6=1.38947655  # asin(60/61)
a7=1.41725254  # asin(84/85)

python train_upper_small.py -a $a1 $a2 $a3 $a4 $a5 -N eq26
python train_upper_small.py -a $a1 $a2 $a3 -b $a4 -B $a5 -N eq27
python train_upper_small.py -a $a1 $a2 $a3 -b $a5 -B $a6 -N eq28
python train_upper_small.py -a $a1 $a2 $a3 $a4 -b $a6 -B $a7 -N eq29
