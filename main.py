from lightning.pytorch.cli import LightningCLI

def main():
    cli = LightningCLI(subclass_mode_model=True, subclass_mode_data=True)

if __name__ == "__main__":
    main()
