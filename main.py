"""Ponto de entrada coordenado do modelo CAPEX."""
from modelo_seguranca_pca_hebb import main as run_security_model


def main():
    """Executa o fluxo completo; efeitos externos ocorrem apenas nesta chamada."""
    run_security_model()


if __name__ == "__main__":
    main()
