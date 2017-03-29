filepath='../data/test_reviews_raw.txt'
savepath='../data/test_reviews_final.txt'
wstype='jieba'
startline=1
addtype='w'

python WSAndPOST.py \
		--filepath=$filepath \
		--savepath=$savepath \
		--wstype=$wstype \
		--startline=$startline \
		--addtype=$addtype
		echo "WS and POST done!"