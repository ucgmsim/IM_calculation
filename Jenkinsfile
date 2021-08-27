pipeline {
    agent any
    stages {
        stage('Install dependencies') {
            steps {
                echo 'Install dependencies on Jenkins server (maybe unnecessary if test runs inside Docker)'

                sh """
                pwd
                env
                source /var/lib/jenkins/py3env/bin/activate
                cd ${env.WORKSPACE}
                pip install -r requirements.txt
                echo ${env.JOB_NAME}
                mkdir -p /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}
                cd /tmp/${env.JOB_NAME}/${env.ghprbActualCommit}
                rm -rf qcore
                git clone https://github.com/ucgmsim/qcore.git
		        mkdir -p ${env.WORKSPACE}/${env.JOB_NAME}/IM/rspectra_calculations/
		        ln -s $HOME/data/testing/${env.JOB_NAME}/rspectra.cpython-37m-x86_64-linux-gnu.so ${env.WORKSPACE}/${env.JOB_NAME}/IM/rspectra_calculations/
                ln -s $HOME/data/testing/${env.JOB_NAME}/sample0 ${env.WORKSPACE}/${env.JOB_NAME}/test
        		cd ${env.WORKSPACE}
		        python travis_setup.py
                """
            }
        }
        stage('Run regression tests') {
            steps {
                echo 'Run pytest'
                sh """
                source /var/lib/jenkins/py3env/bin/activate
                cd ${env.WORKSPACE}/${env.JOB_NAME}
                PYTHONPATH=/tmp/${env.JOB_NAME}/${env.ghprbActualCommit}/qcore
                pytest --black --ignore=test
                cd test
                pytest -vs
                """
            }
        }
    }

    post {
        always {
                echo 'Tear down the environments'
                sh """
                rm -rf /tmp/${env.JOB_NAME}/*
                """
            }
    }
}
