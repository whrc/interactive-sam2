# Interactive Geospatial Labeling Tool for RTS

## 1. Introduction

This project provides a robust, multi-user interactive tool for generating accurate segmentation masks of Retrogressive Thaw Slumps (RTS) from satellite imagery. The tool is designed to run in a Google Colab environment, leveraging a Segment Anything Model (SAM) with few-click prompting to accelerate the creation of high-quality training data for deep learning models.

The primary goal is to streamline the process of refining existing, misaligned polygon labels by allowing a user to interactively generate new, accurate polygons on a stable basemap.

### Key Features

* **Interactive Segmentation**: Uses a pre-trained SAM model to generate segmentation masks from simple positive and negative point prompts.
* **UID-Centric Workflow**: The fundamental unit of work is a unique RTS feature (identified by a UID), allowing for the contextual display of all historical polygons for that feature.
* **Multi-User Support**: A central `manifest.csv` file tracks the labeling progress of each UID, preventing duplicate work and allowing multiple labelers to work in parallel.
* **Direct GCS Integration**: Efficiently loads and processes Planet satellite imagery directly from Google Cloud Storage.
* **Designed for Colab**: The entire workflow is encapsulated in a single Google Colab notebook, providing easy access to free GPU resources for model inference.

***

## 2. How to Use (For Labeling in Google Colab)

This is the primary method for using the tool. **No local installation is required.**

### Steps

1.  **Open the Notebook in Colab**
    * Click the following link to open the main application notebook directly in Google Colab:
        [Open in Colab](https://colab.research.google.com/github/whrc/interactive-sam2/blob/main/notebooks/interactive_label_sam2_(v1_0).ipynb)
    * If the link doesn't work, go to `File > Open notebook` in Colab, select the "GitHub" tab, enter the repository URL (`https://github.com/whrc/interactive-sam2`), and open `notebooks/interactive_label_sam2.ipynb`.

2.  **Run the Setup Cells**
    * Execute the first few cells in the notebook (under the "Environment Setup & Authentication" section).
    * These cells will automatically install the required libraries and clone the project code into the Colab environment.

3.  **Authenticate Your Google Account**
    * You will be prompted to authenticate with your Google account. This is required to grant the notebook secure access to the project's Google Cloud Storage bucket where the images are stored.
    * Follow the on-screen instructions to log in and grant permission.

4.  **Start Labeling**
    * Once the setup and authentication cells have completed successfully, you can run the final cells to launch the interactive user interface and begin labeling.

***

## 3. Developer Setup (For Modifying the Source Code)

This setup is **only for developers** who want to modify the Python source code in the `src/` directory. If you only need to use the tool, please follow the Colab instructions above.

### Prerequisites

1.  **Git**: [Install Git](https://git-scm.com/downloads).
2.  **Git LFS**: [Install Git Large File Storage](https://git-lfs.github.com). This is required to handle the large GeoJSON datasets.
3.  **Anaconda**: [Install Anaconda or Miniconda](https://www.anaconda.com/products/distribution) (Miniconda is recommended).

### Installation Steps

1.  **Clone the Repository**
    Open an Anaconda Prompt or terminal and clone the project.

    ```bash
    git clone [https://github.com/whrc/interactive-sam2.git](https://github.com/whrc/interactive-sam2.git)
    cd interactive-sam2
    ```

2.  **Download Large Files**
    Pull the large data files tracked by Git LFS.

    ```bash
    git lfs pull
    ```

3.  **Create and Activate the Conda Environment**
    This command reads the `environment.yml` file and creates a new, isolated environment named `interactive-sam2` with all the necessary libraries.

    ```bash
    conda env create -f environment.yml
    conda activate interactive-sam2
    ```

You can now edit the files in `src/` and test your changes using the scripts in the `tests/` directory.

***

## 4.Contributing

Contributions to this project are welcome. Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them with clear, descriptive messages.
4.  Push your changes to your fork (`git push origin feature/your-feature-name`).
5.  Create a new Pull Request against the main repository.

## 5. License

This project is licensed under the MIT License - see the `LICENSE` file for details.