pipeline {
    agent any
    environment {
        TEMP_DIR="/tmp/${env.JOB_NAME}/${env.ghprbActualCommit}"
    }
    stages {

        stage('Setting up env') {
            steps {
                echo "[[ Start virtual environment ]]"
                sh """
                    echo "[ Current directory ] : " `pwd`
                    echo "[ Environment Variables ]"
                    env
# Each stage needs custom setting done again. By default /bin/python is used.
                    source /home/qcadmin/py310/bin/activate
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
                    pip install -e qcore
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
                    echo "[ Installing ${env.JOB_NAME} ]"
# full installation is not possible as it takes more than 3.0Gb for building and kills the server
                    pip install -e IM_calculation
                    cd ${env.WORKSPACE}
		            python konno_setup.py
                    echo "[ Linking test data ]"
                    cd ${env.JOB_NAME}/test
                    rm -rf sample0
                    mkdir sample0
                    ln -s /home/qcadmin/data/testing/${env.JOB_NAME}/sample0/input sample0
                    ln -s /home/qcadmin/data/testing/${env.JOB_NAME}/sample0/output sample0
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
