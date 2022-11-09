pipeline {
    agent any
    environment {
        TEMP_DIR="/tmp/${env.JOB_NAME}/${env.ghprbActualCommit}"
    }
    stages {

        stage('Settin up env') {
            steps {
                echo "[[ Start virtual environment ]]"
                sh """
                    echo "[ Current directory ] : " `pwd`
                    echo "[ Environment Variables ] "
                    env
# Each stage needs custom setting done again. By default /bin/python is used.
                    source /var/lib/jenkins/py3env/bin/activate
                    mkdir -p $TEMP_DIR
                    python -m venv $TEMP_DIR/venv
# activate new virtual env
                    source $TEMP_DIR/venv/bin/activate
                    echo "[ Python used ] : " `which python`
                    cd ${env.WORKSPACE}
                    echo "[ Install dependencies ]"
                    pip install -r requirements.txt
                    echo "[ Install qcore ]"
                    cd $TEMP_DIR
                    rm -rf qcore
                    git clone https://github.com/ucgmsim/qcore.git
                    cd qcore
                    python setup.py install --no-data
                """
            }
        }

        stage('Run regression tests') {
            steps {
                echo '[[ Run pytest ]]'
                sh """
# activate virtual environment again
                    source $TEMP_DIR/venv/bin/activate
                    echo "[ Python used ] : " `which python`
                    cd ${env.WORKSPACE}
                    echo "[ Installing ${env.JOB_NAME} ]"
# full installation is not possible as it takes more than 3.0Gb for building and kills the server
#                   python setup.py install
                    python setup.py build_ext --inplace
		            python konno_setup.py
                    echo "[ Linking test data ]"
                    cd ${env.JOB_NAME}/test
                    rm -rf sample0
                    mkdir sample0
                    ln -s $HOME/data/testing/${env.JOB_NAME}/sample0/input sample0
                    ln -s $HOME/data/testing/${env.JOB_NAME}/sample0/output sample0
                    echo "[ Run test now ]"
                    pytest -s
                """
            }
        }
    }

    post {
        always {
                echo 'Tear down the environments'
                sh """
                    rm -rf $TEMP_DIR
                """
            }
    }
}
