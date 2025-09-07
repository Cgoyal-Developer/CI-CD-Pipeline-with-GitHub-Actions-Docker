# 🚀 CI/CD Pipeline with GitHub Actions & Docker (No Cloud Needed)

This project demonstrates how to set up a **CI/CD pipeline** using **GitHub Actions**, **Docker**, and **Docker Hub**, with local deployment using **Minikube** or a local VM — entirely without cloud providers.

---

## 📌 Objective

To build a complete and functional CI/CD pipeline that:

- Builds a Docker image from source code
- Runs automated tests
- Pushes the image to Docker Hub
- Deploys the containerized app to a **local Kubernetes environment (Minikube or VM)**

---

## 🛠️ Tools & Technologies

| Tool           | Purpose                              |
|----------------|--------------------------------------|
| **GitHub Actions** | CI/CD automation                  |
| **Docker**         | Containerization                  |
| **Docker Hub**     | Container registry (free tier)    |
| **Minikube / VM**  | Local Kubernetes deployment       |
| **docker-compose** | Local testing (optional)          |

---

## ✅ Deliverables

- [x] GitHub repository with complete workflows
- [x] Docker image pushed to Docker Hub
- [x] GitHub Actions CI/CD results
- [x] Screenshots of deployed app on Minikube or VM

---

## 📁 Project Structure

.
├── .github/
│ └── workflows/
│ └── ci-cd.yml # GitHub Actions workflow
├── app/
│ ├── index.js # Example Node.js app
│ └── package.json
├── Dockerfile # Docker image definition
├── docker-compose.yml # Local development/testing
├── kubernetes/
│ ├── deployment.yaml
│ └── service.yaml
└── README.md
