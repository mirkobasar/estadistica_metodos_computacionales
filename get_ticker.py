import yfinance as yf
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


tickers = ["^GSPC", "^VIX", "BABA", "EBAY"] 
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=4*365)

try:
    data = yf.download(tickers, start=start_date, end=end_date)
    precios_cierre = data["Close"]
    
    precios_cierre = precios_cierre.rename(columns={
        "^GSPC": "SP500",
        "^VIX": "VIX"
    })

    rendimientos_diarios = np.log(precios_cierre).diff()
    
    rendimientos_diarios = rendimientos_diarios.dropna()
    
    print("Rendimientos diarios en % (primeras 5 filas):")
    print(rendimientos_diarios.head() * 100)

    promedio_muestral = rendimientos_diarios.mean()
    varianza_muestral = rendimientos_diarios.var()
    desvio_estandar = rendimientos_diarios.std()

    estadisticas = pd.DataFrame({
            "Promedio Muestral (%)": promedio_muestral * 100,
            "Varianza Muestral (%^2)": varianza_muestral * (100**2),
            "Desvío Estándar (Volatilidad) (%)": desvio_estandar * 100
        })
  
    pd.options.display.float_format = '{:,.4f}'.format

    print("\nEstadísticas descriptivas:")
    print(estadisticas)

    orden_rendimiento = estadisticas.sort_values(by="Promedio Muestral (%)", ascending=False)
    orden_volatilidad = estadisticas.sort_values(by="Desvío Estándar (Volatilidad) (%)", ascending=False)

    print("\nActivos ordenados por rendimiento promedio (de mayor a menor):")
    print(orden_rendimiento)
    print("\nActivos ordenados por volatilidad (de mayor a menor):")
    print(orden_volatilidad)

    Q1 = rendimientos_diarios.quantile(0.25)
    Q3 = rendimientos_diarios.quantile(0.75)
    IQR = Q3 - Q1
    iqr_ordenado = IQR.sort_values(ascending=False)

    print("\nActivos ordenados por Rango Intercuartil (de mayor a menor) (%):")
    print(iqr_ordenado * 100)

    rendimientos_melt = rendimientos_diarios.melt(var_name='Activo', value_name='Rendimiento Diario')
    rendimientos_melt['Rendimiento Diario'] = rendimientos_melt['Rendimiento Diario'] * 100

    # Gráfico de caja y bigote para los rendimientos diarios ordenados por IQR
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='Activo', 
        y='Rendimiento Diario', 
        data=rendimientos_melt, 
        order=iqr_ordenado.index
    )
    
    plt.ylim(-30, 30)
    plt.title('Gráfico de Caja y Bigote (Zoom a +/- 30%)', fontsize=16)
    plt.ylabel('Rendimiento Diario (%)')
    plt.xlabel('Activos')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    nombre_grafico_rendimientos = "rendimientos_ordenado_caja_y_bigote.png"
    plt.savefig(nombre_grafico_rendimientos)
    print(f"\nGráfico guardado como: {nombre_grafico_rendimientos}")

    # Diagrama de dispersión entre SP500 y VIX
    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        x=rendimientos_diarios["SP500"] * 100,
        y=rendimientos_diarios["VIX"] * 100,
        alpha=0.5
    )

    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.axvline(0, color='red', linestyle='--', linewidth=1)

    plt.title('Diagrama de Dispersión: Rendimientos Diarios SP500 vs. VIX', fontsize=16)
    plt.xlabel('Rendimiento Diario SP500 (%)')
    plt.ylabel('Rendimiento Diario VIX (%)')
    plt.grid(linestyle='--', alpha=0.5)

    nombre_grafico_dispersion = "dispersion_SP500_vs_VIX.png"
    plt.savefig(nombre_grafico_dispersion)
    print(f"Diagrama de Dispersión guardado como: {nombre_grafico_dispersion}")

    matriz_cov = rendimientos_diarios.cov()
    cov_sp500_vix = matriz_cov.loc["SP500", "VIX"]
    print(f"\nLa covarianza muestral entre SP500 y VIX es: {cov_sp500_vix:,.8f}\n")

    # Diagrama de dispersión entre BABA e EBAY
    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        x=rendimientos_diarios["BABA"] * 100,
        y=rendimientos_diarios["EBAY"] * 100,
        alpha=0.5
    )

    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.axvline(0, color='red', linestyle='--', linewidth=1)

    plt.title('Diagrama de Dispersión: Rendimientos Diarios BABA vs. EBAY', fontsize=16)
    plt.xlabel('Rendimiento Diario BABA (%)')
    plt.ylabel('Rendimiento Diario EBAY (%)')
    plt.grid(linestyle='--', alpha=0.5)

    nombre_grafico_dispersion = "dispersion_BABA_vs_EBAY.png"
    plt.savefig(nombre_grafico_dispersion)
    print(f"Diagrama de Dispersión guardado como: {nombre_grafico_dispersion}")

    cov_baba_ebay = matriz_cov.loc["BABA", "EBAY"]
    print(f"\nLa covarianza muestral entre BABA y EBAY es: {cov_baba_ebay:,.8f}\n")

    # Ejercicio Tía Carlota (BABA vs EBAY)

    media_baba = estadisticas.loc["BABA", "Promedio Muestral (%)"]
    media_ebay = estadisticas.loc["EBAY", "Promedio Muestral (%)"]

    pesos_baba = np.linspace(0, 1, 51)
    pesos_ebay = 1 - pesos_baba

    retornos_portafolio = (pesos_baba * media_baba) + (pesos_ebay * media_ebay)

    df_portafolios = pd.DataFrame({
        "Peso en BABA (%)": pesos_baba * 100,
        "Peso en EBAY (%)": pesos_ebay * 100,
        "Rendimiento Esperado (%)": retornos_portafolio
    })

    print("\n51 portafolios generados:")
    print(df_portafolios)
    print("\n")

    mejor_portafolio = df_portafolios.loc[df_portafolios["Rendimiento Esperado (%)"].idxmax()]

    print("El portafolio con el mayor rendimiento esperado es:")
    print(mejor_portafolio)
    print("\n")

    # Ejercicio Tía Carlota (BABA vs EBAY) con riesgo

    var_baba = varianza_muestral.loc["BABA"]
    var_ebay = varianza_muestral.loc["EBAY"]
    cov_baba_ebay = matriz_cov.loc["BABA", "EBAY"]

    var_portafolio = (pesos_baba**2 * var_baba) + (pesos_ebay**2 * var_ebay) + (2 * pesos_baba * pesos_ebay * cov_baba_ebay)

    vol_portafolio = np.sqrt(var_portafolio) * 100

    df_portafolios["Volatilidad (Riesgo) (%)"] = vol_portafolio

    print("\n51 Portafolios (con Riesgo)")
    print(df_portafolios)
    print("\n")

    # Portafolio de minimo riesgo (MVP)
    mvp = df_portafolios.loc[df_portafolios["Volatilidad (Riesgo) (%)"].idxmin()]

    print("\nRECOMENDACION FINAL: Portafolio de Mínima Varianza (Menor Riesgo)")
    print(mvp)

    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        x="Volatilidad (Riesgo) (%)",
        y="Rendimiento Esperado (%)",
        data=df_portafolios,
        label="Portafolios Posibles"
    )

    plt.scatter(
        x=mvp["Volatilidad (Riesgo) (%)"],
        y=mvp["Rendimiento Esperado (%)"],
        color='red',
        s=100,
        edgecolors='black',
        label="Portafolio de Mínima Varianza (MVP)"
    )

    plt.title('Frontera Eficiente (Riesgo vs. Rendimiento) para BABA y EBAY', fontsize=16)
    plt.xlabel('Volatilidad (Riesgo) - Desvío Estándar Diario (%)')
    plt.ylabel('Rendimiento Esperado Diario (%)')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

    nombre_grafico_frontera = "frontera_eficiente_BABA_EBAY.png"
    plt.savefig(nombre_grafico_frontera)
    print(f"\nGráfico de la Frontera Eficiente guardado como: {nombre_grafico_frontera}")

    # Ejercicio Tía Carlota (S&P500 vs VIX)

    media_sp500 = estadisticas.loc["SP500", "Promedio Muestral (%)"]
    media_vix = estadisticas.loc["VIX", "Promedio Muestral (%)"]

    pesos_sp500 = np.linspace(0, 1, 51)
    pesos_vix = 1 - pesos_sp500

    retornos_portafolio = (pesos_sp500 * media_sp500) + (pesos_vix * media_vix)

    df_portafolios = pd.DataFrame({
        "Peso en sp500 (%)": pesos_sp500 * 100,
        "Peso en VIX (%)": pesos_vix * 100,
        "Rendimiento Esperado (%)": retornos_portafolio
    })

    print("\n51 portafolios generados:")
    print(df_portafolios)
    print("\n")

    mejor_portafolio = df_portafolios.loc[df_portafolios["Rendimiento Esperado (%)"].idxmax()]

    print("El portafolio con el mayor rendimiento esperado es:")
    print(mejor_portafolio)
    print("\n")

    # Ejercicio Tía Carlota (S&P500 vs VIX) con riesgo

    var_sp500 = varianza_muestral.loc["SP500"]
    var_vix = varianza_muestral.loc["VIX"]
    cov_sp500_vix = matriz_cov.loc["SP500", "VIX"]

    var_portafolio = (pesos_sp500**2 * var_sp500) + (pesos_vix**2 * var_vix) + (2 * pesos_sp500 * pesos_vix * cov_sp500_vix)

    vol_portafolio = np.sqrt(var_portafolio) * 100

    df_portafolios["Volatilidad (Riesgo) (%)"] = vol_portafolio

    print("\n51 Portafolios (con Riesgo)")
    print(df_portafolios)
    print("\n")

    # Portafolio de minimo riesgo (MVP)
    mvp = df_portafolios.loc[df_portafolios["Volatilidad (Riesgo) (%)"].idxmin()]

    print("\nRECOMENDACION FINAL: Portafolio de Mínima Varianza (Menor Riesgo)")
    print(mvp)

    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        x="Volatilidad (Riesgo) (%)",
        y="Rendimiento Esperado (%)",
        data=df_portafolios,
        label="Portafolios Posibles"
    )

    plt.scatter(
        x=mvp["Volatilidad (Riesgo) (%)"],
        y=mvp["Rendimiento Esperado (%)"],
        color='red',
        s=100,
        edgecolors='black',
        label="Portafolio de Mínima Varianza (MVP)"
    )

    plt.title('Frontera Eficiente (Riesgo vs. Rendimiento) para SP500 y VIX', fontsize=16)
    plt.xlabel('Volatilidad (Riesgo) - Desvío Estándar Diario (%)')
    plt.ylabel('Rendimiento Esperado Diario (%)')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

    nombre_grafico_frontera = "frontera_eficiente_SP500_VIX.png"
    plt.savefig(nombre_grafico_frontera)
    print(f"\nGráfico de la Frontera Eficiente guardado como: {nombre_grafico_frontera}")


except Exception as e:
    print(f"Ocurrió un error: {e}")