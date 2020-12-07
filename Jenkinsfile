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
                mkdir -p /tmp/${env.JOB_NAME}
                cd /tmp/${env.JOB_NAME}
                rm -rf qcore
                git clone https://github.com/ucgmsim/qcore.git
                pip install --no-deps ./qcore/
		mkdir -p ${env.WORKSPACE}/${env.JOB_NAME}/IM/rspectra_calculations/
		mkdir -p ${env.WORKSPACE}/${env.JOB_NAME}/IM/iesdr_calculation/
		ln -s $HOME/data/testing/${env.JOB_NAME}/rspectra.cpython-37m-x86_64-linux-gnu.so ${env.WORKSPACE}/${env.JOB_NAME}/IM/rspectra_calculations/
		ln -s $HOME/data/testing/${env.JOB_NAME}/Burks_Baker_2013_elastic_inelastic.cpython-37m-x86_64-linux-gnu.so ${env.WORKSPACE}/${env.JOB_NAME}/IM/iesdr_calculation/
                ln -s $HOME/data/testing/${env.JOB_NAME}/sample0 ${env.WORKSPACE}/${env.JOB_NAME}/test
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
                docker container prune -f
                """
            }
    }
}
