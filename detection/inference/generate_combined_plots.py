import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np



def process(args):
    with open("./outputs/classical_params.pkl", "rb") as f:
        classical_params = pickle.load(f)

    with open("./outputs/unet_params.pkl", "rb") as f:
        unet_params = pickle.load(f)

    with open("./outputs/checkerboard_params.pkl", "rb") as f:
        checkerboard_params = pickle.load(f)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title('Fx')
    axs[0, 1].set_title('Fy')
    axs[1, 0].set_title('Cx')
    axs[1, 1].set_title('Cy')


    Fx_classical = [x[1][0,0] for x in classical_params]
    Fx_unet      = [x[1][0,0] for x in unet_params]
    Fx_checkboard= [x[1][0,0] for x in checkerboard_params]

    axs[0,0].hist(Fx_classical,density=True, bins=100, alpha=0.5, label='classical')
    axs[0,0].hist(Fx_unet,density=True, bins=100, alpha=0.5, label='unet')
    axs[0,0].hist(Fx_checkboard,density=True, bins=100, alpha=0.5, label='chessboard')
    axs[0,0].legend(loc='upper right')


    Fy_classical = [x[1][1,1] for x in classical_params]
    Fy_unet      = [y[1][1,1] for y in unet_params]
    Fy_checkboard= [x[1][1,1] for x in checkerboard_params]

    axs[0,1].hist(Fy_classical,density=True, bins=100, alpha=0.5, label='classical')
    axs[0,1].hist(Fy_unet,density=True, bins=100, alpha=0.5, label='unet')
    axs[0,1].hist(Fy_checkboard,density=True, bins=100, alpha=0.5, label='chessboard')
    axs[0,1].legend(loc='upper right')



    Cx_classical = [x[1][0,2] for x in classical_params]
    Cx_unet      = [y[1][0,2] for y in unet_params]
    Cx_checkboard= [x[1][0,2] for x in checkerboard_params]

    axs[1,0].hist(Cx_classical,density=True, bins=100, alpha=0.5, label='classical')
    axs[1,0].hist(Cx_unet,density=True, bins=100, alpha=0.5, label='unet')
    axs[1,0].hist(Cx_checkboard,density=True, bins=100, alpha=0.5, label='chessboard')
    axs[1,0].legend(loc='upper right')


    Cy_classical = [x[1][1,2] for x in classical_params]
    Cy_unet      = [y[1][1,2] for y in unet_params]
    Cy_checkboard= [x[1][1,2] for x in checkerboard_params]

    axs[1,1].hist(Cy_classical,density=True, bins=100, alpha=0.5, label='classical')
    axs[1,1].hist(Cy_unet,density=True, bins=100, alpha=0.5, label='unet')
    axs[1,1].hist(Cy_checkboard,density=True, bins=100, alpha=0.5, label='chessboard')
    axs[1,1].legend(loc='upper right')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--visualize",type = bool,
                        help="Draw images for debugging", default = True)
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main();
