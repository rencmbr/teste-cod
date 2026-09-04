import unittest
import numpy as np
import scipy.linalg as la

from src.fem_vnmm_hybrid_2d import (
    gerar_malha_hibrida_cavidade,
    montar_matrizes_hibridas_fem_vnmm,
    resolver_autovalores_hibrido_fem_vnmm
)


class TestHybridFemVnmm(unittest.TestCase):
    """
    Testes unitários para o solver híbrido acoplado FEM de Aresta 2D + VNMM 2D (P1).
    """
    
    def test_geracao_malha_hibrida(self):
        """Valida a geração das malhas particionadas e o alinhamento na interface."""
        Lx = np.pi
        Ly = np.pi
        Nex_fem = 6
        Ney = 10
        Nx_vnmm = 7
        Ny_vnmm = 11
        
        dados_fem, dados_vnmm = gerar_malha_hibrida_cavidade(
            Lx=Lx, Ly=Ly, frac_fem=0.5, Nex_fem=Nex_fem, Ney=Ney,
            Nx_vnmm=Nx_vnmm, Ny_vnmm=Ny_vnmm
        )
        
        # 1. Verifica número de arestas de interface no FEM
        n_interface_fem = len(dados_fem['interface_eids'])
        self.assertEqual(n_interface_fem, Ney)
        
        # 2. Verifica que os nós de interface do VNMM têm a mesma quantidade
        n_interface_vnmm = np.sum(dados_vnmm['is_interface'])
        self.assertEqual(n_interface_vnmm, Ney)
        
        # 3. Verifica que os vetores na interface são verticais unitários t = [0, 1]
        idx_intf = np.where(dados_vnmm['is_interface'])[0]
        for idx in idx_intf:
            vec = dados_vnmm['vectors'][idx]
            self.assertAlmostEqual(vec[0], 0.0, places=6)
            self.assertAlmostEqual(vec[1], 1.0, places=6)
            # Ponto x deve ser igual a x_int
            self.assertAlmostEqual(dados_vnmm['coords'][idx, 0], dados_fem['x_int'], places=6)
            
    def test_simetria_matrizes_hibridas(self):
        """Valida a simetria estrita das matrizes globais acopladas."""
        dados_fem, dados_vnmm = gerar_malha_hibrida_cavidade(
            Nex_fem=4, Ney=6, Nx_vnmm=5, Ny_vnmm=7, frac_fem=0.5
        )
        
        K_glob, M_glob, info_dofs = montar_matrizes_hibridas_fem_vnmm(
            dados_fem, dados_vnmm, Ncx_vnmm=4, Ncy_vnmm=6, s_div_vnmm=6.0
        )
        
        K_dense = K_glob.toarray()
        M_dense = M_glob.toarray()
        
        # Verifica simetria
        erro_sim_K = np.max(np.abs(K_dense - K_dense.T))
        erro_sim_M = np.max(np.abs(M_dense - M_dense.T))
        
        self.assertLess(erro_sim_K, 1e-12)
        self.assertLess(erro_sim_M, 1e-12)
        
        # Verifica que M é estritamente definida positiva
        evals_M = la.eigvalsh(M_dense)
        self.assertTrue(np.all(evals_M > 0.0))
        
    def test_acuracia_modos_fundamentais(self):
        """Valida a acurácia dos modos físicos TE10, TE01, TE11 no solver híbrido."""
        res = resolver_autovalores_hibrido_fem_vnmm(
            Lx=np.pi, Ly=np.pi, frac_fem=0.5,
            Nex_fem=8, Ney=12, Nx_vnmm=9, Ny_vnmm=13,
            Ncx_vnmm=8, Ncy_vnmm=10, s_div_vnmm=6.0,
            num_autovalores=3
        )
        
        # Modos TE10, TE01, TE11
        erros_kc = res['erros_kc_pct'][:3]
        for e in erros_kc:
            self.assertLess(e, 5.0, f"Erro de kc ({e:.2f}%) excedeu o limite de 5.0%")
            
        self.assertLess(res['erro_medio_kc_pct'], 4.0)

    def test_hibrido_com_malhas_aleatorias(self):
        """Valida o funcionamento e acurácia do acoplamento híbrido com malhas e direções aleatórias."""
        res = resolver_autovalores_hibrido_fem_vnmm(
            Lx=np.pi, Ly=np.pi, frac_fem=0.5,
            Nex_fem=6, Ney=10, Nx_vnmm=7, Ny_vnmm=11,
            Ncx_vnmm=6, Ncy_vnmm=8, s_div_vnmm=6.0,
            pontos_por_dir=2,
            tipo_interior_vnmm="aleatorio",
            jitter_frac_fem=0.25,
            jitter_frac_vnmm=0.25,
            num_autovalores=3,
            seed=42
        )
        
        # Verifica que o erro médio dos 3 primeiros modos permanece contido abaixo de 5%
        self.assertLess(res['erro_medio_kc_pct'], 5.0)
        self.assertEqual(len(res['autovalores_numericos']), 3)


if __name__ == "__main__":
    unittest.main()
