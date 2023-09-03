java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomTreeGenerator -o 0 -u 20 -v 100) -d (generators.RandomTreeGenerator -r 99 -i 10 -o 0 -u 20 -v 100) -p 50000 -w 20000) -f Gradual_RT_10.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomTreeGenerator -o 0 -u 20 -v 100) -d (generators.RandomTreeGenerator -r 99 -i 20 -o 0 -u 20 -v 100) -p 50000 -w 20000) -f Gradual_RT_20.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomTreeGenerator -o 0 -u 20 -v 100) -d (generators.RandomTreeGenerator -r 99 -i 30 -o 0 -u 20 -v 100) -p 50000 -w 20000) -f Gradual_RT_30.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomTreeGenerator -o 0 -u 20 -v 100) -d (generators.RandomTreeGenerator -r 99 -i 40 -o 0 -u 20 -v 100) -p 50000 -w 20000) -f Gradual_RT_40.arff -m 100000"

java -cp moa.jar moa.DoTask "WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomTreeGenerator -o 0 -u 20 -v 100) -d (generators.RandomTreeGenerator -r 99 -i 50 -o 0 -u 20 -v 100) -p 50000 -w 20000) -f Gradual_RT_50.arff -m 100000"
