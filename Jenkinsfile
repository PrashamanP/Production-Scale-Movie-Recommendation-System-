pipeline {
  agent any

  options {
    timestamps()
    ansiColor('xterm')
    timeout(time: 10, unit: 'MINUTES')
  }

  environment {
    VENV = ".venv"
    PY   = "python3"
  }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    // Post a "pending" PR status at the very start (only on PRs)
    stage('Set PR status: pending') {
      when { changeRequest() }
      steps {
        script {
          try {
            setGitHubPullRequestStatus(
              state: 'PENDING',                 // REQUIRED
              context: 'ci/jenkins',            // Shown on the PR
              message: "Build #${env.BUILD_NUMBER} started",
              sha: env.GIT_COMMIT,              // commit to update
              url: env.BUILD_URL                // link back to this run
            )
          } catch (e) {
            echo "PR status (pending) not set: ${e}"
          }
        }
      }
    }

    stage('Set up Python') {
      steps {
        sh '''
          set -e
          ${PY} -V
          if [ ! -d "${VENV}" ]; then ${PY} -m venv ${VENV}; fi
          . ${VENV}/bin/activate
          pip install --upgrade pip
        '''
      }
    }

    stage('Install deps') {
      steps {
        sh '''
          set -e
          . ${VENV}/bin/activate
          test -f requirements.txt && pip install -r requirements.txt || true
          pip install pytest pytest-cov
        '''
      }
    }

    stage('Unit tests') {
      steps {
        sh '''
          set -e
          . ${VENV}/bin/activate
          mkdir -p reports
          pytest -q --maxfail=1 --disable-warnings \
            --junitxml=reports/junit.xml \
            --cov=src --cov-config=.coveragerc \
            --cov-report=xml:reports/coverage.xml \
            --cov-report=term-missing \
            --cov-fail-under=0 \
            tests
        '''
      }
      post {
        always {
          archiveArtifacts artifacts: 'reports/**', fingerprint: true, onlyIfSuccessful: false
          junit 'reports/junit.xml'
          // Cobertura XML produced by pytest-cov
          recordCoverage tools: [[parser: 'COBERTURA', pattern: 'reports/coverage.xml']], failOnError: false
        }
      }
    }
  }

  post {
    success {
      script {
        if (env.CHANGE_ID) {
          try {
            setGitHubPullRequestStatus(
              state: 'SUCCESS',
              context: 'ci/jenkins',
              message: "Build #${env.BUILD_NUMBER} passed",
              sha: env.GIT_COMMIT,
              url: env.BUILD_URL
            )
          } catch (e) {
            echo "PR status (success) not set: ${e}"
          }
        }
      }
    }
    failure {
      script {
        if (env.CHANGE_ID) {
          try {
            setGitHubPullRequestStatus(
              state: 'FAILURE',
              context: 'ci/jenkins',
              message: "Build #${env.BUILD_NUMBER} failed",
              sha: env.GIT_COMMIT,
              url: env.BUILD_URL
            )
          } catch (e) {
            echo "PR status (failure) not set: ${e}"
          }
        }
      }
    }
    unstable {
      script {
        if (env.CHANGE_ID) {
          try {
            setGitHubPullRequestStatus(
              state: 'ERROR',
              context: 'ci/jenkins',
              message: "Build #${env.BUILD_NUMBER} unstable",
              sha: env.GIT_COMMIT,
              url: env.BUILD_URL
            )
          } catch (e) {
            echo "PR status (unstable) not set: ${e}"
          }
        }
      }
    }
  }
}