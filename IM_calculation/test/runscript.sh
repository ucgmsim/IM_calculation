# random change to test master testing

DIRS='test_*'
curdir=$PWD
for DIR in $DIRS
do
cd ${curdir}/${DIR}
pytest --junitxml ${DIR}.xml
cp -r ./${DIR}.xml /home/tester/.jenkins/workspace/im_calc_tests/${DIR}.xml
done
LOG=/home/tester/.jenkins/jobs/im_calc_tests/builds/$BUILD_NUMBER/log
cp ${LOG} ${LOG}.html
sed -i 's/$/<br>/' ${LOG}.html
