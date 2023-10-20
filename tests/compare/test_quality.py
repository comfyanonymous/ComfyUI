import datetime
import numpy as np
import os
from PIL import Image
import pytest
from pytest import fixture
from typing import Tuple, List

from cv2 import imread, cvtColor, COLOR_BGR2RGB
from skimage.metrics import structural_similarity as ssim


"""
This test suite compares images in 2 directories by file name
The directories are specified by the command line arguments --baseline_dir and --test_dir

"""
# ssim: Structural Similarity Index
# Returns a tuple of (ssim, diff_image)
def ssim_score(img0: np.ndarray, img1: np.ndarray) -> Tuple[float, np.ndarray]:
    score, diff = ssim(img0, img1, channel_axis=-1, full=True)
    # rescale the difference image to 0-255 range
    diff = (diff * 255).astype("uint8")
    return score, diff
    
# Metrics must return a tuple of (score, diff_image)
METRICS = {"ssim": ssim_score}
METRICS_PASS_THRESHOLD = {"ssim": 0.95}


class TestCompareImageMetrics:
    @fixture(scope="class")
    def test_file_names(self, args_pytest):
        test_dir = args_pytest['test_dir']
        fnames = self.gather_file_basenames(test_dir)  
        yield fnames
        del fnames

    @fixture(scope="class", autouse=True)
    def teardown(self, args_pytest):
        yield
        # Runs after all tests are complete
        # Aggregate output files into a grid of images
        baseline_dir = args_pytest['baseline_dir']
        test_dir = args_pytest['test_dir']
        img_output_dir = args_pytest['img_output_dir']
        metrics_file = args_pytest['metrics_file']

        grid_dir = os.path.join(img_output_dir, "grid")
        os.makedirs(grid_dir, exist_ok=True)

        for metric_dir in METRICS.keys():
            metric_path = os.path.join(img_output_dir, metric_dir)
            for file in os.listdir(metric_path):
                if file.endswith(".png"):
                    score = self.lookup_score_from_fname(file, metrics_file)
                    image_file_list = []
                    image_file_list.append([
                                            os.path.join(baseline_dir, file),
                                            os.path.join(test_dir, file),
                                            os.path.join(metric_path, file)
                                            ])
                    # Create grid
                    image_list = [[Image.open(file) for file in files] for files in image_file_list]
                    grid = self.image_grid(image_list)
                    grid.save(os.path.join(grid_dir, f"{metric_dir}_{score:.3f}_{file}"))
    
    # Tests run for each baseline file name
    @fixture()
    def fname(self, baseline_fname):
        yield baseline_fname
        del baseline_fname
    
    def test_directories_not_empty(self, args_pytest):
        baseline_dir = args_pytest['baseline_dir']
        test_dir = args_pytest['test_dir']
        assert len(os.listdir(baseline_dir)) != 0, f"Baseline directory {baseline_dir} is empty"
        assert len(os.listdir(test_dir)) != 0, f"Test directory {test_dir} is empty"

    def test_dir_has_all_matching_metadata(self, fname, test_file_names, args_pytest):
        # Check that all files in baseline_dir have a file in test_dir with matching metadata
        baseline_file_path = os.path.join(args_pytest['baseline_dir'], fname)
        file_paths = [os.path.join(args_pytest['test_dir'], f) for f in test_file_names]
        file_match = self.find_file_match(baseline_file_path, file_paths)
        assert file_match is not None, f"Could not find a file in {args_pytest['test_dir']} with matching metadata to {baseline_file_path}"

    # For a baseline image file, finds the corresponding file name in test_dir and 
    # compares the images using the metrics in METRICS
    @pytest.mark.parametrize("metric", METRICS.keys())
    def test_pipeline_compare(
        self,
        args_pytest,
        fname,
        test_file_names,
        metric,
    ):
        baseline_dir = args_pytest['baseline_dir']
        test_dir = args_pytest['test_dir']
        metrics_output_file = args_pytest['metrics_file']
        img_output_dir = args_pytest['img_output_dir']
        
        baseline_file_path = os.path.join(baseline_dir, fname)

        # Find file match
        file_paths = [os.path.join(test_dir, f) for f in test_file_names]
        test_file = self.find_file_match(baseline_file_path, file_paths)

        # Run metrics
        sample_baseline = self.read_img(baseline_file_path)
        sample_secondary = self.read_img(test_file)
        
        score, metric_img = METRICS[metric](sample_baseline, sample_secondary)
        metric_status = score > METRICS_PASS_THRESHOLD[metric]

        # Save metric values
        with open(metrics_output_file, 'a') as f:
            run_info = os.path.splitext(fname)[0]
            metric_status_str = "PASS ✅" if metric_status else "FAIL ❌"
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"| {date_str} | {run_info} | {metric} | {metric_status_str} | {score} | \n")

        # Save metric image
        metric_img_dir = os.path.join(img_output_dir, metric)
        os.makedirs(metric_img_dir, exist_ok=True)
        output_filename = f'{fname}'
        Image.fromarray(metric_img).save(os.path.join(metric_img_dir, output_filename))

        assert score > METRICS_PASS_THRESHOLD[metric]

    def read_img(self, filename: str) -> np.ndarray:
        cvImg = imread(filename)
        cvImg = cvtColor(cvImg, COLOR_BGR2RGB)
        return cvImg

    def image_grid(self, img_list: list[list[Image.Image]]):
        # imgs is a 2D list of images
        # Assumes the input images are a rectangular grid of equal sized images
        rows = len(img_list)
        cols = len(img_list[0])

        w, h = img_list[0][0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        
        for i, row in enumerate(img_list):
            for j, img in enumerate(row):
                grid.paste(img, box=(j*w, i*h))
        return grid

    def lookup_score_from_fname(self,
                                fname: str,
                                metrics_output_file: str
        ) -> float:
        fname_basestr = os.path.splitext(fname)[0]
        with open(metrics_output_file, 'r') as f:
            for line in f:
                if fname_basestr in line:
                    score = float(line.split('|')[5])
                    return score
        raise ValueError(f"Could not find score for {fname} in {metrics_output_file}")

    def gather_file_basenames(self, directory: str):
        files = []
        for file in os.listdir(directory):
            if file.endswith(".png"):
                files.append(file)
        return files

    def read_file_prompt(self, fname:str) -> str:
        # Read prompt from image file metadata
        img = Image.open(fname)
        img.load()
        return img.info['prompt']
    
    def find_file_match(self, baseline_file: str, file_paths: List[str]):
        # Find a file in file_paths with matching metadata to baseline_file
        baseline_prompt = self.read_file_prompt(baseline_file)

        # Do not match empty prompts
        if baseline_prompt is None or baseline_prompt == "":
            return None

        # Find file match
        # Reorder test_file_names so that the file with matching name is first
        # This is an optimization because matching file names are more likely 
        # to have matching metadata if they were generated with the same script
        basename = os.path.basename(baseline_file)
        file_path_basenames = [os.path.basename(f) for f in file_paths]
        if basename in file_path_basenames:
            match_index = file_path_basenames.index(basename)
            file_paths.insert(0, file_paths.pop(match_index))

        for f in file_paths:
            test_file_prompt = self.read_file_prompt(f)
            if baseline_prompt == test_file_prompt:
                return f