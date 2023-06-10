from lightning.pytorch.cli import LightningCLI

# 2.0.0+cu117
# 0.10.0+cu111
def main():
    cli = LightningCLI(subclass_mode_model=True, subclass_mode_data=True)

if __name__ == "__main__":
    main()
