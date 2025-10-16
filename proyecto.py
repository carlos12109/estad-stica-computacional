"""
Sistema de Simulación Computacional
Proyecto Parcial de Primera Unidad
Autor: VIDMAN RUIS ROQUE MAMANI
Fecha: Octubre 2025

============================================================================
                        DESCRIPCIÓN GENERAL DEL PROYECTO
============================================================================

Este proyecto es una aplicación de escritorio desarrollada en Python con Tkinter
que sirve como una herramienta educativa para la simulación computacional y la
estadística. Permite a los usuarios explorar conceptos clave a través de una
interfaz gráfica interactiva.

El sistema está organizado en los siguientes módulos principales:

1.  **MÓDULO: Generador de Variables Aleatorias (`GeneradorPropio`, `GeneradorAleatorio`)**
    - Implementa un generador de números pseudoaleatorios desde cero utilizando el
      algoritmo Blum Blum Shub (BBS) para garantizar la independencia de librerías
      externas de aleatoriedad.
    - Proporciona métodos para generar variables aleatorias de diversas
      distribuciones (discretas y continuas) utilizando técnicas como la
      transformada inversa y el algoritmo de Box-Muller.

2.  **MÓDULO: Pruebas de Bondad de Ajuste (`PruebasBondadAjuste`)**
    - Ofrece métodos estadísticos (Chi-Cuadrado y Kolmogorov-Smirnov) para
      evaluar si un conjunto de datos se ajusta a una distribución teórica.

3.  **MÓDULO: Simulaciones Monte Carlo (`MonteCarlo`)**
    - Contiene la lógica para ejecutar simulaciones basadas en aleatoriedad para
      estimar resultados de problemas complejos, como la estimación de Pi, la
      ruina del jugador y sistemas de colas.

4.  **INTERFAZ GRÁFICA (`AplicacionSimulacion`)**
    - Es la clase principal que construye la GUI, gestiona los eventos del
      usuario y conecta todos los módulos para presentar una experiencia cohesiva.
      Utiliza hilos (`threading`) para ejecutar simulaciones largas sin congelar
      la interfaz.
"""
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import threading
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
import time
import math
import sys

# ============================================================================
# MÓDULO: Generador de Variables Aleatorias
# ============================================================================

class GeneradorPropio:
    """
    Implementa un generador de números pseudoaleatorios Blum Blum Shub (BBS).

    Este generador no depende de ninguna librería externa de aleatoriedad (como `random`
    o `numpy.random`). Su funcionamiento se basa en la fórmula de recurrencia:
    X_{n+1} = (X_n)^2 mod M, donde M es el producto de dos números primos grandes.

    Atributos:
        m (int): El módulo M, calculado como el producto de dos primos p y q.
        estado (int): El estado interno actual del generador (X_n).
        _z1_cache (float or None): Un caché para almacenar el segundo valor generado
            por la transformada de Box-Muller, optimizando la generación de números
            normales.
    """
    def __init__(self, semilla):
        """Inicializa el generador BBS con una semilla dada."""
        # --- CAMBIO RADICAL: Usar el algoritmo Blum Blum Shub ---
        # Se eligen dos primos p y q tal que p ≡ 3 (mod 4) y q ≡ 3 (mod 4).
        # Estos no son "grandes", pero son suficientes para una demostración.
        p = 499  # 499 % 4 = 3
        q = 503  # 503 % 4 = 3
        self.m = p * q  # Módulo M = 250997

        # La semilla (estado inicial) debe ser coprima con M y no ser 0 o 1.
        # Una forma simple es s^2 mod M, donde s es la semilla del usuario.
        estado_inicial = semilla % self.m
        if estado_inicial <= 1:
            estado_inicial = 3 # Un valor inicial seguro por defecto
        self.estado = (estado_inicial * estado_inicial) % self.m

        # --- MEJORA: Caché para la transformada de Box-Muller ---
        self._z1_cache = None

    def _next_int(self):
        """Genera el siguiente entero en la secuencia pseudo-aleatoria."""
        self.estado = pow(self.estado, 2, self.m) # Equivalente a (self.estado ** 2) % self.m
        return self.estado

    def random(self):
        """Genera un número flotante pseudo-aleatorio en el rango [0, 1)."""
        return self._next_int() / self.m

    def uniform(self, a, b):
        """Genera un número flotante en un rango [a, b)."""
        return a + (b - a) * self.random()

    def normal(self, media, std):
        """Genera un número normal usando la transformada de Box-Muller.

        Este método es eficiente porque genera dos números normales estándar (z0 y z1)
        a la vez a partir de dos números uniformes (u1 y u2). El primer número (z0)
        se devuelve inmediatamente, y el segundo (z1) se guarda en un caché para
        la siguiente llamada, evitando así la mitad de los cálculos en la próxima
        invocación.

        Args:
            media (float): La media (μ) de la distribución normal.
            std (float): La desviación estándar (σ) de la distribución normal.

        Returns:
            float: Un número aleatorio que sigue una distribución N(media, std).
        """
        if self._z1_cache is not None:
            # Si hay un valor en el caché, usarlo y limpiar el caché.
            z1 = self._z1_cache
            self._z1_cache = None
            return z1 * std + media

        # Generar dos números uniformes y aplicar la transformada de Box-Muller.
        u1 = 1.0 - self.random()
        u2 = 1.0 - self.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        self._z1_cache = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        return z0 * std + media

class GeneradorAleatorio:
    """
    Clase de alto nivel para generar variables aleatorias de diversas distribuciones.

    Utiliza una instancia de `GeneradorPropio` para obtener los números uniformes
    base y luego los transforma para generar las distribuciones deseadas.

    Atributos:
        semilla (int): La semilla utilizada para inicializar el generador.
        generador_propio (GeneradorPropio): La instancia del generador BBS subyacente.
    """
    
    def __init__(self, semilla=None):
        """Inicializa el generador. Si no se proporciona semilla, se crea una robusta."""
        if semilla is None:
            # --- MEJORA: Generación de semilla más robusta ---
            # Se combinan múltiples fuentes de entropía para una semilla más impredecible:
            # 1. Tiempo del sistema en nanosegundos para alta precisión.
            # 2. ID del proceso actual, que cambia en cada ejecución.
            # 3. ID del objeto en memoria, que es único para cada instancia.
            # Estos valores se combinan con operaciones a nivel de bits (XOR) para mezclar su aleatoriedad.
            self.semilla = (time.time_ns() ^ os.getpid() ^ id(self)) % (2**32)
        else:
            self.semilla = semilla
        # ---Usa el generador propio en lugar de NumPy ---
        self.generador_propio = GeneradorPropio(self.semilla)
    
    def bernoulli(self, p, n):
        """Genera `n` variables aleatorias de Bernoulli con probabilidad de éxito `p`."""
        if not 0 <= p <= 1:
            raise ValueError("p debe estar entre 0 y 1")
        return np.array([1 if self.generador_propio.random() < p else 0 for _ in range(n)])
    
    def binomial(self, n_trials, p, size):
        """Genera `size` variables aleatorias Binomiales."""
        if not 0 <= p <= 1:
            raise ValueError("p debe estar entre 0 y 1")
        # Simula una variable binomial como la suma de 'n_trials' bernoulli
        resultados = []
        for _ in range(size):
            exitos = sum(1 for _ in range(n_trials) if self.generador_propio.random() < p)
            resultados.append(exitos)
        return np.array(resultados)
    
    def geometrica(self, p, size):
        """Genera `size` variables aleatorias Geométricas usando la transformada inversa."""
        if not 0 < p <= 1:
            raise ValueError("p debe estar entre 0 y 1")
        # Usando transformada inversa: floor(log(1-U) / log(1-p))
        return np.array([math.floor(math.log(1.0 - self.generador_propio.random()) / math.log(1 - p)) + 1 for _ in range(size)])

    def poisson(self, lambda_param, size):
        """Genera `size` variables aleatorias de Poisson usando el algoritmo de Knuth."""
        if lambda_param <= 0:
            raise ValueError("Lambda debe ser mayor que 0")
        # ---Algoritmo de Knuth más robusto para generar variables Poisson ---
        resultados = []
        L = math.exp(-lambda_param)
        for _ in range(size):
            k = 0
            p = 1.0
            while p > L:
                k += 1
                # Usar 1.0 - random() para evitar que p se vuelva 0 y cause un bucle infinito.
                p *= (1.0 - self.generador_propio.random())
            resultados.append(k - 1)
        return np.array(resultados)
    
    def uniforme(self, a, b, size):
        """Genera `size` variables aleatorias Uniformes en el rango [a, b)."""
        if a >= b:
            raise ValueError("a debe ser menor que b")
        return np.array([self.generador_propio.uniform(a, b) for _ in range(size)])
    
    def exponencial(self, lambda_param, size):
        """Genera `size` variables aleatorias Exponenciales usando la transformada inversa."""
        if lambda_param <= 0:
            raise ValueError("Lambda debe ser mayor que 0")
        # Usando transformada inversa: -ln(1-U) / lambda
        return np.array([-math.log(1.0 - self.generador_propio.random()) / lambda_param for _ in range(size)])
    
    def normal(self, media, desv_std, size):
        """Genera `size` variables aleatorias Normales."""
        if desv_std <= 0:
            raise ValueError("Desviación estándar debe ser mayor que 0")
        # --- MEJORA: Aprovechar el caché de Box-Muller ---
        # Limpiar el caché al inicio de una nueva generación para consistencia.
        self.generador_propio._z1_cache = None
        resultados = [self.generador_propio.normal(media, desv_std) for _ in range(size)]
        self.generador_propio._z1_cache = None # Limpiar al final también.
        return np.array(resultados)

# ============================================================================
# MÓDULO: Pruebas de Bondad de Ajuste
# ============================================================================

class PruebasBondadAjuste:
    """
    Proporciona métodos estáticos para realizar pruebas de bondad de ajuste.

    Estas pruebas evalúan qué tan bien un conjunto de datos de muestra se ajusta a una
    distribución de probabilidad teórica.
    """
    
    @staticmethod
    def _agrupar_bins_chi_cuadrado(observado, esperado, bins):
        """Agrupa intervalos para la prueba Chi-Cuadrado.

        Este método auxiliar combina los intervalos (bins) de los extremos para
        asegurar que todas las frecuencias esperadas sean >= 5. Este es un
        requisito estadístico para la validez de la prueba Chi-Cuadrado.

        Returns:
            tuple: Una tupla con las frecuencias observadas, esperadas y los
                   límites de los intervalos, todos agrupados según la regla
                   de frecuencia esperada mínima.
        """
        # Copiar para no modificar los arrays originales
        obs = list(observado)
        esp = list(esperado)
        bns = list(bins)

        # Agrupar desde el final hacia el principio
        i = len(esp) - 1
        while i > 0:
            if esp[i] < 5:
                esp[i-1] += esp[i]
                obs[i-1] += obs[i]
                esp.pop(i)
                obs.pop(i)
                bns.pop(i) # Eliminar el límite del bin que se fusionó
            i -= 1
            
        # Agrupar desde el principio hacia el final
        i = 0
        while i < len(esp) - 1:
            if esp[i] < 5:
                esp[i+1] += esp[i]
                obs[i+1] += obs[i]
                esp.pop(i)
                obs.pop(i)
                bns.pop(i+1) # Eliminar el límite del bin que se fusionó
            else:
                i += 1
                
        return np.array(obs), np.array(esp), np.array(bns)

    @staticmethod
    def chi_cuadrado(datos, distribucion, params):
        """Realiza la prueba Chi-cuadrado para bondad de ajuste.

        Args:
            datos (np.array): La muestra de datos a probar.
            distribucion (str): El nombre de la distribución a probar ('normal',
                                'exponencial', 'uniforme').
            params (tuple): Los parámetros de la distribución teórica.

        Returns:
            dict: Un diccionario con los resultados de la prueba, incluyendo el
                  estadístico chi2, el valor p, los grados de libertad, y las
                  frecuencias observadas y esperadas.
        """
        # Crear bins para datos observados
        n_bins = min(20, max(10, len(np.unique(datos)) // 5))
        observado, bins = np.histogram(datos, bins=n_bins, density=False)
        
        # Calcular frecuencias esperadas
        if distribucion == 'normal':
            media, std = params
            esperado = []
            for i in range(len(bins)-1):
                prob = stats.norm.cdf(bins[i+1], media, std) - stats.norm.cdf(bins[i], media, std)
                esperado.append(prob * len(datos))
        elif distribucion == 'exponencial':
            lambda_p = params[0]
            esperado = []
            for i in range(len(bins)-1):
                prob = stats.expon.cdf(bins[i+1], scale=1/lambda_p) - stats.expon.cdf(bins[i], scale=1/lambda_p)                
                esperado.append(prob * len(datos))
        elif distribucion == 'uniforme':
            a, b = params
            esperado = []
            for i in range(len(bins)-1):
                prob = stats.uniform.cdf(bins[i+1], a, b-a) - stats.uniform.cdf(bins[i], a, b-a)
                esperado.append(prob * len(datos))
        
        esperado = np.array(esperado)

        # --- MEJORA: Agrupar bins con frecuencias esperadas bajas (< 5) ---
        # Se reemplaza la asignación directa por un método estadísticamente más robusto.
        observado_agrupado, esperado_agrupado, bins_agrupados = PruebasBondadAjuste._agrupar_bins_chi_cuadrado(observado, esperado, bins)
        
        # Si después de agrupar quedan muy pocos intervalos, la prueba no es válida.
        if len(observado_agrupado) < 2:
            raise ValueError("No hay suficientes intervalos (>1) después de agrupar las frecuencias esperadas. La prueba Chi-Cuadrado no se puede realizar. Intente con una muestra de datos más grande o con mayor variabilidad.")

        # Calcular estadístico chi-cuadrado
        chi2 = np.sum((observado_agrupado - esperado_agrupado)**2 / esperado_agrupado)
        # Los grados de libertad se calculan con el número de bins *después* de agrupar.
        gl = max(1, len(observado_agrupado) - 1 - len(params))
        p_valor = 1 - stats.chi2.cdf(chi2, gl)
        
        return {
            'estadistico': chi2,
            'p_valor': p_valor,
            'grados_libertad': gl,
            'observado': observado_agrupado,
            'esperado': esperado_agrupado,
            'bins': bins_agrupados
        }
    
    @staticmethod
    def kolmogorov_smirnov(datos, distribucion, params):
        """Realiza la prueba Kolmogorov-Smirnov (K-S) para bondad de ajuste.

        Esta prueba compara la función de distribución acumulada empírica (ECDF)
        de los datos con la función de distribución acumulada teórica (CDF).

        Args:
            datos (np.array): La muestra de datos a probar.
            distribucion (str): El nombre de la distribución a probar.
            params (tuple): Los parámetros de la distribución teórica.

        Returns:
            dict: Un diccionario con el estadístico de prueba 'D' y el 'p_valor'.
        """
        if distribucion == 'normal':
            media, std = params
            estadistico, p_valor = stats.kstest(datos, 'norm', args=(media, std))
        elif distribucion == 'exponencial':
            lambda_p = params[0]
            estadistico, p_valor = stats.kstest(datos, 'expon', args=(0, 1/lambda_p))
        elif distribucion == 'uniforme':
            a, b = params
            estadistico, p_valor = stats.kstest(datos, 'uniform', args=(a, b-a))
        
        return {
            'estadistico': estadistico,
            'p_valor': p_valor
        }

# ============================================================================
# MÓDULO: Simulaciones Monte Carlo
# ============================================================================

class MonteCarlo:
    """
    Proporciona métodos estáticos para ejecutar diversas simulaciones Monte Carlo.

    Las simulaciones utilizan el `GeneradorPropio` para asegurar que los resultados
    se basan en el generador de números aleatorios implementado en este proyecto.
    Cada método de simulación puede aceptar un `progress_callback` para reportar
    el progreso a la interfaz de usuario.
    """
    # --- CAMBIO RADICAL: Usar el generador propio en lugar de NumPy ---
    # Se crea una instancia con una semilla basada en el tiempo para que cada ejecución sea diferente.
    semilla_mc = (time.time_ns() ^ os.getpid()) % (2**32)
    rng = GeneradorPropio(semilla_mc)
    
    @staticmethod
    def estimar_pi(n_simulaciones, progress_callback=None):
        """Estima el valor de Pi usando el método de Monte Carlo.

        La simulación genera puntos aleatorios en un cuadrado de 2x2 centrado en
        el origen y cuenta cuántos caen dentro de un círculo unitario inscrito.
        La estimación de Pi se calcula como 4 * (puntos_dentro / puntos_totales).

        Args:
            n_simulaciones (int): El número de puntos aleatorios a generar.
            progress_callback (function, optional): Una función para reportar el
                progreso de la simulación.

        Returns:
            dict: Un diccionario con los resultados, incluyendo la estimación de Pi.
        """
        dentro_circulo = 0
        puntos_x = []
        puntos_y = []
        colores = []
        historial_pi = []  # Para guardar la estimación en cada paso

        # Para no actualizar la GUI en cada iteración, lo que sería muy lento
        update_interval = max(1, n_simulaciones // 100)
        
        for i in range(1, n_simulaciones + 1):
            x = MonteCarlo.rng.uniform(-1, 1)
            y = MonteCarlo.rng.uniform(-1, 1)
            
            puntos_x.append(x)
            puntos_y.append(y)
            
            if x**2 + y**2 <= 1:
                dentro_circulo += 1
                colores.append('blue')
            else:
                colores.append('red')
            
            # Calcular y guardar la estimación actual de pi
            pi_actual = 4 * dentro_circulo / i
            historial_pi.append(pi_actual)

            # Reportar progreso
            if progress_callback and (i % update_interval == 0 or i == n_simulaciones):
                progress_callback(i, n_simulaciones)
        
        pi_estimado = historial_pi[-1]
        
        return {
            'pi_estimado': pi_estimado,
            'puntos_x': puntos_x,
            'puntos_y': puntos_y,
            'colores': colores,
            'dentro': dentro_circulo,
            'total': n_simulaciones,
            'historial_pi': historial_pi
        }
    
    @staticmethod
    def ruina_jugador(capital_inicial, prob_ganar, apuesta, objetivo, n_simulaciones, progress_callback=None):
        """Simula el problema de la ruina del jugador.

        Modela la fortuna de un jugador que apuesta repetidamente una cantidad fija
        hasta que alcanza un capital objetivo o se queda sin dinero (la ruina).

        Args:
            capital_inicial (float): El dinero con el que empieza el jugador.
            prob_ganar (float): La probabilidad de ganar una sola apuesta.
            apuesta (float): La cantidad apostada en cada jugada.
            objetivo (float): El capital que el jugador desea alcanzar.
            n_simulaciones (int): El número de veces que se repetirá el juego completo.

        Returns:
            dict: Un diccionario con las probabilidades de ruina y éxito, y la duración promedio.
        """
        ruinas = 0
        exitos = 0
        duraciones = []

        # Para no actualizar la GUI en cada iteración
        update_interval = max(1, n_simulaciones // 100)
        
        for i in range(n_simulaciones):
            capital = capital_inicial
            pasos = 0
            
            while capital > 0 and capital < objetivo and pasos < 10000:
                if MonteCarlo.rng.random() < prob_ganar:
                    capital += apuesta
                else:
                    capital -= apuesta
                pasos += 1
            
            duraciones.append(pasos)
            if capital <= 0:
                ruinas += 1
            elif capital >= objetivo:
                exitos += 1
            
            # Reportar progreso
            if progress_callback and ((i + 1) % update_interval == 0 or (i + 1) == n_simulaciones):
                progress_callback(i + 1, n_simulaciones)
        
        return {
            'prob_ruina': ruinas / n_simulaciones,
            'prob_exito': exitos / n_simulaciones,
            'duracion_promedio': np.mean(duraciones),
            'duraciones': duraciones
        }
    
    @staticmethod
    def cola_mm1(lambda_llegada, mu_servicio, tiempo_simulacion, progress_callback=None):
        """Simula un sistema de colas M/M/1.

        Este modelo representa un sistema con un único servidor donde los tiempos
        entre llegadas de clientes y los tiempos de servicio siguen una distribución
        exponencial (proceso de Poisson).

        Args:
            lambda_llegada (float): La tasa promedio de llegada de clientes (λ).
            mu_servicio (float): La tasa promedio de servicio del servidor (μ).
            tiempo_simulacion (float): La duración total de la simulación.

        Returns:
            dict: Un diccionario con métricas de rendimiento, como el tiempo de
                  espera promedio y un registro de eventos detallado.
        """
        clientes_atendidos = 0
        tiempos_espera = []
        event_log = [] # MEJORA: Log de eventos para visualización
        
        # Generar eventos de llegada
        eventos_llegada = []
        t = 0
        while t < tiempo_simulacion:
            # Generar tiempo entre llegadas (Exponencial)
            t += -math.log(1.0 - MonteCarlo.rng.random()) / lambda_llegada
            if t < tiempo_simulacion:
                eventos_llegada.append(t)
        
        # Simular sistema
        tiempo_libre_servidor = 0
        total_eventos = len(eventos_llegada)
        # Para no actualizar la GUI en cada iteración
        update_interval = max(1, total_eventos // 100)
        
        for i, tiempo_llegada in enumerate(eventos_llegada):
            # Generar tiempo de servicio (Exponencial)
            tiempo_servicio = -math.log(1.0 - MonteCarlo.rng.random()) / mu_servicio
            
            # --- CORRECCIÓN: Lógica de la simulación de colas ---
            if tiempo_llegada >= tiempo_libre_servidor:
                # Servidor está libre
                tiempo_espera = 0
                inicio_servicio = tiempo_llegada
            else:
                # Servidor está ocupado, el cliente debe esperar.
                tiempo_espera = tiempo_libre_servidor - tiempo_llegada
                inicio_servicio = tiempo_libre_servidor

            fin_servicio = inicio_servicio + tiempo_servicio
            tiempo_libre_servidor = fin_servicio

            tiempos_espera.append(tiempo_espera)
            clientes_atendidos += 1

            event_log.append({
                'cliente_id': i + 1,
                'llegada': tiempo_llegada,
                'inicio_servicio': inicio_servicio,
                'fin_servicio': fin_servicio,
                'espera': tiempo_espera,
                'duracion_servicio': tiempo_servicio
            })

            # Reportar progreso
            if progress_callback and ((i + 1) % update_interval == 0 or (i + 1) == total_eventos):
                progress_callback(i + 1, total_eventos)
        
        return {
            'tiempo_espera_promedio': np.mean(tiempos_espera) if tiempos_espera else 0,
            'clientes_atendidos': clientes_atendidos,
            'tiempos_espera': tiempos_espera,
            'event_log': event_log,
            'clientes_en_cola_promedio': np.mean([te for te in tiempos_espera if te > 0]) if any(te > 0 for te in tiempos_espera) else 0
        }

# ============================================================================
# INTERFAZ GRÁFICA PRINCIPAL
# ============================================================================

class AplicacionSimulacion:
    """
    Clase principal de la aplicación que construye y gestiona la interfaz gráfica.

    Esta clase utiliza `tkinter.ttk` para crear una interfaz moderna con pestañas.
    Cada pestaña corresponde a una funcionalidad principal del sistema:
    - Generación de Variables Aleatorias
    - Pruebas de Bondad de Ajuste
    - Simulaciones Monte Carlo
    - Pestañas informativas (Fórmulas, Ayuda, etc.)
    """
    
    def __init__(self, root):
        """Inicializa la ventana principal y todos los componentes de la GUI."""
        self.root = root
        self.root.title("Sistema de Simulación Computacional")
        self.root.geometry("1200x800")

        # --- CORRECCIÓN: Ruta absoluta para el ícono ---
        # Esto asegura que el ícono se encuentre sin importar desde dónde se ejecute el script.
        try:
            # Determina la ruta base (funciona tanto para script como para ejecutable de PyInstaller)
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            icon_path = os.path.join(base_path, 'VDM-LOGO.ico')
            self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Advertencia: No se pudo cargar el ícono 'VDM-LOGO.ico'. Error: {e}")
        
        # --- Paleta de Colores y Fuentes Mejorada ---
        self.bg_color = '#f1f5f9'          # Gris claro para el fondo
        self.frame_bg_color = '#ffffff'    # Blanco para los frames internos
        self.text_color = '#0f172a'        # Azul oscuro (casi negro) para texto principal
        self.subtext_color = '#475569'     # Gris azulado para texto secundario
        self.accent_color = '#2563eb'      # Azul vibrante para acentos y botones
        self.accent_fg_color = '#ffffff'   # Blanco para texto sobre acentos
        self.success_color = '#10b981'     # Verde para éxito
        self.error_color = '#ef4444'       # Rojo para errores

        self.font_main = ('Segoe UI', 10)
        self.font_bold = ('Segoe UI', 10, 'bold')
        self.font_title = ('Segoe UI', 14, 'bold')
        self.font_header = ('Segoe UI', 11, 'bold')
        self.font_code = ('Consolas', 10)

        self.root.configure(bg=self.bg_color)

        # --- Configuración de Estilos TTK ---
        style = ttk.Style()
        style.theme_use('clam')

        # Estilo general para widgets
        style.configure('TNotebook', background=self.bg_color)
        style.configure('TNotebook.Tab', padding=[15, 8], font=self.font_bold, background='#e2e8f0', foreground=self.subtext_color)
        style.map('TNotebook.Tab', background=[('selected', self.frame_bg_color)], foreground=[('selected', self.accent_color)])

        style.configure('TFrame', background=self.bg_color)
        style.configure('White.TFrame', background=self.frame_bg_color) # Frame blanco

        style.configure('TLabelframe', background=self.bg_color, foreground=self.text_color, bordercolor=self.subtext_color)
        style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.text_color, font=self.font_bold)

        style.configure('TLabel', background=self.bg_color, foreground=self.text_color, font=self.font_main)
        style.configure('Accent.TLabel', foreground=self.accent_color, background=self.bg_color, font=self.font_title)

        # Estilo para botones
        style.configure('TButton', font=self.font_bold, padding=8, borderwidth=0)
        style.configure('Accent.TButton', background=self.accent_color, foreground=self.accent_fg_color)
        style.map('Accent.TButton', background=[('active', '#1d4ed8')])
        # --- MEJORA: Estilo para botones secundarios ---
        style.configure('Secondary.TButton', background='#e2e8f0', foreground=self.subtext_color)
        style.map('Secondary.TButton', background=[('active', '#cbd5e1')])


        # Crear notebook (pestañas)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Crear pestañas
        self.crear_pestaña_generacion()
        self.crear_pestaña_bondad_ajuste()
        self.crear_pestaña_monte_carlo()
        self.crear_pestaña_formulas()
        self.crear_pestaña_resumen_teorico()
        self.crear_pestaña_ayuda()
        self.acerca_de()
        # Bandera para controlar la ejecución de Monte Carlo
        self.is_mc_running = False

    def acerca_de(self):
        """Crea la pestaña 'Acerca del Software' con información detallada."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Acerca del Software")

        # Centrar contenido
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        content_frame = ttk.Frame(frame)
        content_frame.grid(row=0, column=0, pady=20, padx=20)

        # Título principal
        title_label = ttk.Label(content_frame, text="Sistema de Simulación Computacional",
                                  style='Accent.TLabel',
                                  font=('Segoe UI', 24, 'bold'))
        title_label.pack(pady=(0, 10))

        # Subtítulo
        subtitle_label = ttk.Label(content_frame,
                                     text="Una herramienta para el aprendizaje de la estadística y la simulación.",
                                     font=('Arial', 12))
        subtitle_label.pack(pady=(0, 25))
        
        # --- Objetivo del Software ---
        objetivo_frame = ttk.LabelFrame(content_frame, text="🎯 Objetivo del Software", padding=15)
        objetivo_frame.pack(pady=10, padx=20, fill='x')
        objetivo_text = """Este sistema ha sido desarrollado como una herramienta educativa interactiva. Su principal objetivo es facilitar la comprensión de conceptos fundamentales de la estadística computacional y la simulación a través de la experimentación directa. Los usuarios pueden generar datos, realizar pruebas estadísticas y visualizar los resultados de simulaciones complejas de una manera intuitiva y gráfica."""
        objetivo_label = ttk.Label(objetivo_frame, text=objetivo_text, justify='left', font=self.font_main, wraplength=1000)
        objetivo_label.pack(fill='x')

        # --- Tecnologías Utilizadas ---
        tech_frame = ttk.LabelFrame(content_frame, text="🛠️ Tecnologías Utilizadas", padding=15)
        tech_frame.pack(pady=10, padx=20, fill='x')
        tech_text = """•  Python: Lenguaje principal de desarrollo.
•  Tkinter (ttk): Para la construcción de la interfaz gráfica de usuario.
•  NumPy: Para el manejo eficiente de arrays y cálculos numéricos.
•  Matplotlib: Para la generación de gráficos y visualizaciones.
•  SciPy: Para la implementación de pruebas estadísticas avanzadas."""
        tech_label = ttk.Label(tech_frame, text=tech_text, justify='left', font=self.font_main)
        tech_label.pack(fill='x')

        # --- Créditos ---
        credits_frame = ttk.LabelFrame(content_frame, text="Créditos", padding=15)
        credits_frame.pack(pady=10, padx=20, fill='x')

        credits_info = {
            "Autor:": "VIDMAN RUIS ROQUE MAMANI",
            "Universidad:": "Universidad Nacional del Altiplano, Puno",
            "Escuela Profesional:": "Ingeniería Estadística e Informática",
            "Curso:": "Estadística Computacional",
            "Docente:": "Dr. Edgardo Yapo Quispe"
        }

        for i, (key, value) in enumerate(credits_info.items()):
            # Etiqueta (ej. "Autor:")
            key_label = ttk.Label(credits_frame, text=key, font=self.font_bold, background=self.bg_color)
            key_label.grid(row=i, column=0, sticky='e', padx=(0, 5), pady=2)
            
            # Valor (ej. "VIDMAN RUIS ROQUE MAMANI")
            value_label = ttk.Label(credits_frame, text=value, font=self.font_main, background=self.bg_color)
            value_label.grid(row=i, column=1, sticky='w', padx=(5, 0), pady=2)

    def crear_pestaña_formulas(self):
        """Crea la pestaña para mostrar fórmulas importantes."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Fórmulas")

        # Scrollbar para el contenido
        scroll_y = ttk.Scrollbar(frame)
        scroll_y.pack(side='right', fill='y')

        text_formulas = tk.Text(frame, wrap='word', yscrollcommand=scroll_y.set,
                                font=self.font_main, bg=self.frame_bg_color, fg=self.text_color,
                                padx=20, pady=15, spacing1=5, spacing3=12, relief='flat', borderwidth=0)
        text_formulas.pack(fill='both', expand=True)
        scroll_y.config(command=text_formulas.yview)

        # --- MEJORA: Definir etiquetas de estilo para el texto de fórmulas ---
        text_formulas.tag_configure('title', font=('Segoe UI', 18, 'bold'), foreground=self.accent_color, justify='center', spacing3=20)
        text_formulas.tag_configure('h1', font=self.font_title, foreground=self.text_color, spacing1=15, spacing3=8, background='#e2e8f0', lmargin1=5, lmargin2=5)
        text_formulas.tag_configure('h2', font=self.font_header, foreground=self.accent_color, spacing1=10)
        text_formulas.tag_configure('formula', font=self.font_code, lmargin1=20, lmargin2=20, spacing1=5, spacing3=5)
        text_formulas.tag_configure('desc', font=self.font_main, lmargin1=20, lmargin2=20)
        text_formulas.tag_configure('separator', font=('Consolas', 8), justify='center', foreground=self.subtext_color)

        # --- Contenido de las fórmulas con formato ---
        text_formulas.insert(tk.END, "Fórmulas Teóricas Utilizadas\n", 'title')

        # --- Sección de Distribuciones ---
        text_formulas.insert(tk.END, "\n Distribuciones de Probabilidad \n", 'h1')

        text_formulas.insert(tk.END, "Distribución Normal (PDF)\n", 'h2')
        text_formulas.insert(tk.END, "         1        -(x-μ)²\n f(x) = ───── e^(───────)\n        σ√(2π)       2σ²\n", 'formula')
        text_formulas.insert(tk.END, "\u03BC: media, \u03C3: desviación estándar\n", 'desc')

        text_formulas.insert(tk.END, "Distribución Exponencial (PDF)\n", 'h2')
        text_formulas.insert(tk.END, "f(x) = λe⁻λx  para x ≥ 0\n", 'formula')
        text_formulas.insert(tk.END, "\u03BB: tasa de ocurrencia\n", 'desc')

        text_formulas.insert(tk.END, "Distribución Uniforme (PDF)\n", 'h2')
        text_formulas.insert(tk.END, "       ⎧ 1/(b-a)   para a ≤ x ≤ b\nf(x) = ⎨\n       ⎩ 0         en otro caso\n", 'formula')
        text_formulas.insert(tk.END, "a: límite inferior, b: límite superior\n", 'desc')

        text_formulas.insert(tk.END, "Distribución de Poisson (PMF)\n", 'h2')
        text_formulas.insert(tk.END, "         λᵏ e⁻λ\nP(X=k) = ────────\n           k!\n", 'formula')
        text_formulas.insert(tk.END, "\u03BB: número promedio de eventos, k: número de ocurrencias\n", 'desc')

        text_formulas.insert(tk.END, "Distribución Binomial (PMF)\n", 'h2')
        text_formulas.insert(tk.END, "P(X=k) = C(n,k) pᵏ (1-p)ⁿ⁻ᵏ\n", 'formula')
        text_formulas.insert(tk.END, "n: número de ensayos, p: probabilidad de éxito, k: número de éxitos\n", 'desc')

        text_formulas.insert(tk.END, "\n" + "─" * 120 + "\n", 'separator')

        # --- Sección de Pruebas de Ajuste ---
        text_formulas.insert(tk.END, "\n Pruebas de Bondad de Ajuste \n", 'h1')

        text_formulas.insert(tk.END, "Estadístico Chi-Cuadrado (\u03C7\u00B2)\n", 'h2')
        text_formulas.insert(tk.END, "       (Oᵢ - Eᵢ)²\n χ² = Σ ──────────\n           Eᵢ\n", 'formula')
        text_formulas.insert(tk.END, "O\u1D62: frecuencia observada en el intervalo i\nE\u1D62: frecuencia esperada en el intervalo i\n", 'desc')

        text_formulas.insert(tk.END, "Estadístico Kolmogorov-Smirnov (D)\n", 'h2')
        text_formulas.insert(tk.END, "D = sup | Fₙ(x) - F(x) |\n      x\n", 'formula')
        text_formulas.insert(tk.END, "F\u2099(x): función de distribución empírica\nF(x): función de distribución teórica\n", 'desc')

        text_formulas.insert(tk.END, "\n" + "─" * 120 + "\n", 'separator')

        # --- Sección de Simulaciones ---
        text_formulas.insert(tk.END, "\n Simulaciones Monte Carlo \n", 'h1')

        text_formulas.insert(tk.END, "Ruina del Jugador (Probabilidad de Ruina)\n", 'h2')
        text_formulas.insert(tk.END, "Si p ≠ 0.5:        (q/p)ᴬ - (q/p)ᶻ\n         P(ruina) = ───────────────\n                      (q/p)ᴬ - 1\n", 'formula')
        text_formulas.insert(tk.END, "Si p = 0.5:  P(ruina) = 1 - Z/A\n", 'formula')
        text_formulas.insert(tk.END, "p: prob. de ganar, q: 1-p, Z: capital inicial, A: capital objetivo\n", 'desc')

        text_formulas.insert(tk.END, "Sistema de Colas M/M/1\n", 'h2')
        text_formulas.insert(tk.END, "ρ = λ/μ (Factor de utilización)\n", 'formula')
        text_formulas.insert(tk.END, "      ρ²\nLq = ───── (Clientes promedio en cola)\n    1 - ρ\n", 'formula')
        text_formulas.insert(tk.END, "      Lq\nWq = ─── (Tiempo promedio en cola)\n      λ\n", 'formula')
        text_formulas.insert(tk.END, "\u03BB: tasa de llegada, \u03BC: tasa de servicio\n", 'desc')

        text_formulas.config(state='disabled') # Hacer el texto de solo lectura

    def crear_pestaña_resumen_teorico(self):
        """Crea una pestaña con el resumen teórico de los modelos de simulación."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Resumen Teórico")

        scroll_y = ttk.Scrollbar(frame)
        scroll_y.pack(side='right', fill='y')

        text_resumen = tk.Text(frame, wrap='word', yscrollcommand=scroll_y.set,
                               font=self.font_main, bg=self.frame_bg_color, fg=self.text_color,
                               padx=20, pady=15, spacing1=5, spacing3=12, relief='flat', borderwidth=0)
        text_resumen.pack(fill='both', expand=True)
        scroll_y.config(command=text_resumen.yview)

        # Usar los mismos estilos que las pestañas de Fórmulas y Ayuda
        text_resumen.tag_configure('title', font=('Segoe UI', 18, 'bold'), foreground=self.accent_color, justify='center', spacing3=20)
        text_resumen.tag_configure('h1', font=self.font_title, foreground=self.text_color, spacing1=15, spacing3=8, background='#e2e8f0', lmargin1=5, lmargin2=5)
        text_resumen.tag_configure('h2', font=self.font_header, foreground=self.accent_color, spacing1=10)
        text_resumen.tag_configure('desc', font=self.font_main, lmargin1=20, lmargin2=20)
        text_resumen.tag_configure('separator', font=('Consolas', 8), justify='center', foreground=self.subtext_color)

        # --- Contenido del Resumen Teórico ---
        text_resumen.insert(tk.END, "Fundamentos de los Modelos de Simulación\n", 'title')

        # --- Estimación de Pi ---
        text_resumen.insert(tk.END, "\n 1. Estimación de π (Pi) por Monte Carlo \n", 'h1')
        text_resumen.insert(tk.END, "Concepto General\n", 'h2')
        text_resumen.insert(tk.END, "Este método utiliza la aleatoriedad para aproximar el valor de π. Se basa en una relación geométrica simple: la proporción entre el área de un círculo y el área del cuadrado que lo circunscribe.\n", 'desc')
        text_resumen.insert(tk.END, "Procedimiento\n", 'h2')
        text_resumen.insert(tk.END, "1.  Se inscribe un círculo de radio 'r' dentro de un cuadrado de lado '2r'.\n"
                                     "2.  Se generan miles de puntos aleatorios (x, y) dentro de los límites del cuadrado.\n"
                                     "3.  Se cuenta cuántos de estos puntos caen dentro del círculo (condición: x² + y² ≤ r²).\n"
                                     "4.  La relación (puntos_dentro / puntos_totales) es una aproximación de la relación de las áreas (Área_Círculo / Área_Cuadrado).\n"
                                     "5.  Sabiendo que Área_Círculo = πr² y Área_Cuadrado = (2r)² = 4r², se puede despejar π:  π ≈ 4 * (puntos_dentro / puntos_totales).\n", 'desc')

        text_resumen.insert(tk.END, "\n" + "─" * 120 + "\n", 'separator')

        # --- Ruina del Jugador ---
        text_resumen.insert(tk.END, "\n 2. Ruina del Jugador \n", 'h1')
        text_resumen.insert(tk.END, "Concepto General\n", 'h2')
        text_resumen.insert(tk.END, "Es un problema clásico de la teoría de la probabilidad que modela una 'caminata aleatoria' con dos barreras absorbentes. Simula la fortuna de un jugador que apuesta repetidamente hasta que alcanza un objetivo o se queda sin dinero (la ruina).\n", 'desc')
        text_resumen.insert(tk.END, "Elementos del Modelo\n", 'h2')
        text_resumen.insert(tk.END, "•  Capital Inicial (z): El dinero con el que empieza el jugador.\n"
                                     "•  Probabilidad de Ganar (p): La probabilidad de ganar una sola apuesta.\n"
                                     "•  Objetivo (A): La cantidad de dinero que el jugador desea alcanzar.\n"
                                     "•  Barreras Absorbentes: El juego termina si el capital llega a 0 (ruina) o a A (éxito).\n", 'desc')

        text_resumen.insert(tk.END, "\n" + "─" * 120 + "\n", 'separator')

        # --- Sistema de Colas M/M/1 ---
        text_resumen.insert(tk.END, "\n 3. Sistema de Colas M/M/1 \n", 'h1')
        text_resumen.insert(tk.END, "Notación de Kendall\n", 'h2')
        text_resumen.insert(tk.END, "El modelo M/M/1 se describe mediante la notación de Kendall (A/B/s):\n"
                                     "•  M (Llegadas): Las llegadas de clientes siguen un proceso de Poisson, lo que significa que el tiempo entre llegadas consecutivas sigue una distribución Exponencial (M de Markoviano).\n"
                                     "•  M (Servicio): Los tiempos de servicio también siguen una distribución Exponencial.\n"
                                     "•  1 (Servidores): Hay un único servidor para atender a los clientes.\n", 'desc')
        text_resumen.insert(tk.END, "Conceptos Clave\n", 'h2')
        text_resumen.insert(tk.END, "•  Tasa de Llegada (λ): El número promedio de clientes que llegan por unidad de tiempo.\n"
                                     "•  Tasa de Servicio (μ): El número promedio de clientes que un servidor puede atender por unidad de tiempo.\n"
                                     "•  Condición de Estabilidad: Para que la cola no crezca indefinidamente, el sistema debe ser estable. Esto ocurre cuando la tasa de servicio es mayor que la de llegada (μ > λ).\n", 'desc')

        text_resumen.config(state='disabled') # Hacer el texto de solo lectura

    def crear_pestaña_ayuda(self):
        """Crea la pestaña de Ayuda con una guía de uso detallada."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Ayuda")

        scroll_y = ttk.Scrollbar(frame)
        scroll_y.pack(side='right', fill='y')

        text_guia = tk.Text(frame, wrap='word', yscrollcommand=scroll_y.set,
                            font=self.font_main, bg=self.frame_bg_color, fg=self.text_color,
                            padx=20, pady=15, spacing1=5, spacing3=12, relief='flat', borderwidth=0)
        text_guia.pack(fill='both', expand=True)
        scroll_y.config(command=text_guia.yview)

        # Usar los mismos estilos que la pestaña de Fórmulas para consistencia
        text_guia.tag_configure('title', font=('Segoe UI', 18, 'bold'), foreground=self.accent_color, justify='center', spacing3=20)
        text_guia.tag_configure('h1', font=self.font_title, foreground=self.text_color, spacing1=15, spacing3=8, background='#e2e8f0', lmargin1=5, lmargin2=5)
        text_guia.tag_configure('h2', font=self.font_header, foreground=self.accent_color, spacing1=10)
        text_guia.tag_configure('desc', font=self.font_main, lmargin1=20, lmargin2=20)
        text_guia.tag_configure('code', font=self.font_code, background='#f8fafc', foreground='#475569', lmargin1=20, lmargin2=20)
        text_guia.tag_configure('separator', font=('Consolas', 8), justify='center', foreground=self.subtext_color)

        # --- Contenido de la Guía ---
        text_guia.insert(tk.END, "Guía de Uso del Sistema\n", 'title')

        # --- Generación de Variables Aleatorias ---
        text_guia.insert(tk.END, "\n 1. Generación de Variables Aleatorias \n", 'h1')
        text_guia.insert(tk.END, "Esta pestaña permite crear conjuntos de datos que siguen una distribución de probabilidad específica.\n", 'desc')
        
        text_guia.insert(tk.END, "Pasos a seguir:\n", 'h2')
        text_guia.insert(tk.END, "1.  Configuración: Seleccione la distribución deseada, el número de muestras a generar y, opcionalmente, una semilla para que los resultados sean reproducibles. Si deja la semilla en blanco, se usará una aleatoria.\n", 'desc')
        text_guia.insert(tk.END, "2.  Parámetros: Ingrese los parámetros requeridos para la distribución seleccionada (ej. media y desviación estándar para la Normal).\n", 'desc')
        text_guia.insert(tk.END, "3.  Acciones: Presione 'Generar' para crear los datos. Use los otros botones para visualizar un histograma, copiar el reporte o los datos, o exportar el reporte a un archivo .txt.\n", 'desc')

        # --- Pruebas de Bondad de Ajuste ---
        text_guia.insert(tk.END, "\n" + "─" * 120 + "\n", 'separator')
        text_guia.insert(tk.END, "\n 2. Pruebas de Bondad de Ajuste \n", 'h1')
        text_guia.insert(tk.END, "Aquí puede verificar si un conjunto de datos se ajusta a una distribución teórica (Normal, Exponencial o Uniforme).\n", 'desc')

        text_guia.insert(tk.END, "Pasos a seguir:\n", 'h2')
        text_guia.insert(tk.END, "1.  Entrada de Datos: Cargue sus datos desde un archivo .txt, ingréselos manualmente o use los datos de la pestaña anterior ('Usar datos generados').\n", 'desc')
        text_guia.insert(tk.END, "2.  Configuración de la Prueba: Elija la distribución teórica que desea probar, el método (Chi-Cuadrado o K-S) y el nivel de significancia (alfa, α), comúnmente 0.05.\n", 'desc')
        text_guia.insert(tk.END, "3.  Realizar Prueba: Presione el botón para ejecutar el análisis.\n", 'desc')

        text_guia.insert(tk.END, "Interpretación del Resultado:\n", 'h2')
        text_guia.insert(tk.END, "El sistema comparará el 'valor p' con su nivel de significancia 'α'.\n", 'desc')
        text_guia.insert(tk.END, "  •  Si p > α: No se rechaza la hipótesis nula (H₀). La evidencia sugiere que los datos SÍ se ajustan a la distribución probada.\n", 'code')
        text_guia.insert(tk.END, "  •  Si p ≤ α: Se rechaza la hipótesis nula (H₀). La evidencia sugiere que los datos NO se ajustan a la distribución probada.\n", 'code')

        # --- Método de Monte Carlo ---
        text_guia.insert(tk.END, "\n" + "─" * 120 + "\n", 'separator')
        text_guia.insert(tk.END, "\n 3. Método de Monte Carlo \n", 'h1')
        text_guia.insert(tk.END, "Esta sección utiliza la aleatoriedad para simular y estimar resultados de problemas complejos.\n", 'desc')

        text_guia.insert(tk.END, "Pasos a seguir:\n", 'h2')
        text_guia.insert(tk.END, "1.  Seleccionar Problema: Elija una de las simulaciones disponibles en el menú desplegable.\n", 'desc')
        text_guia.insert(tk.END, "2.  Parámetros: Ajuste los parámetros específicos para la simulación elegida (ej. número de simulaciones, capital inicial, etc.).\n", 'desc')
        text_guia.insert(tk.END, "3.  Ejecutar Simulación: Presione el botón para iniciar. Las simulaciones intensivas pueden tardar unos segundos y mostrarán una barra de progreso.\n", 'desc')

        text_guia.insert(tk.END, "Interpretación del Resultado:\n", 'h2')
        text_guia.insert(tk.END, "El reporte mostrará los resultados de la simulación (ej. valor estimado de π, probabilidad de ruina). En muchos casos, también se mostrará el resultado teórico para que pueda comparar la precisión de la simulación.\n", 'desc')

        text_guia.config(state='disabled') # Hacer el texto de solo lectura
        
    def crear_pestaña_generacion(self):
        """Crea la pestaña de generación de variables aleatorias"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Generación de Variables Aleatorias")
        
        # --- MEJORA: Crear un frame superior para colocar la configuración y los parámetros en paralelo ---
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill='x', padx=10, pady=5)

        # Frame de configuración
        config_frame = ttk.LabelFrame(top_frame, text="Configuración", padding=15)
        config_frame.pack(side='left', fill='y', padx=(0, 5), anchor='n')
        
        # --- MEJORA: Diccionario para organizar distribuciones ---
        self.distribuciones = {
            "Discreta": ['Bernoulli', 'Binomial', 'Geometrica', 'Poisson'],
            "Continua": ['Uniforme', 'Exponencial', 'Normal']
        }

        # Semilla
        ttk.Label(config_frame, text="Semilla (dejar vacío para automática):").grid(row=0, column=0, sticky='w', pady=5, padx=5)
        self.semilla_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.semilla_var, width=25).grid(row=0, column=1, pady=5, padx=5)
        
        # --- MEJORA: Selector de tipo de distribución (Discreta/Continua) ---
        ttk.Label(config_frame, text="Tipo de Distribución:").grid(row=1, column=0, sticky='w', pady=5, padx=5)
        self.dist_tipo_var = tk.StringVar(value="Discreta")
        tipo_combo = ttk.Combobox(config_frame, textvariable=self.dist_tipo_var, width=23, state='readonly')
        tipo_combo['values'] = list(self.distribuciones.keys())
        tipo_combo.grid(row=1, column=1, pady=5, padx=5)
        tipo_combo.bind('<<ComboboxSelected>>', self.actualizar_lista_distribuciones)

        # Distribución específica
        ttk.Label(config_frame, text="Distribución:").grid(row=2, column=0, sticky='w', pady=5, padx=5)
        self.dist_var = tk.StringVar()
        dist_combo = ttk.Combobox(config_frame, textvariable=self.dist_var, width=23, state='readonly')
        self.dist_combo = dist_combo # Guardar referencia para actualizarla
        dist_combo.grid(row=2, column=1, pady=5, padx=5)
        dist_combo.bind('<<ComboboxSelected>>', self.actualizar_parametros_dist)
        
        # Número de muestras
        ttk.Label(config_frame, text="Número de muestras:").grid(row=3, column=0, sticky='w', pady=5, padx=5)
        self.n_muestras_var = tk.StringVar(value="1000")
        ttk.Entry(config_frame, textvariable=self.n_muestras_var, width=25).grid(row=3, column=1, pady=5, padx=5)
        
        # --- MEJORA: Frame de parámetros al lado del de configuración ---
        self.params_frame = ttk.LabelFrame(top_frame, text="Parámetros de la Distribución", padding=15)
        self.params_frame.pack(side='left', fill='both', expand=True, padx=(5, 0))
        
        self.param_entries = {}
        self.actualizar_parametros_dist(None)
        self.actualizar_lista_distribuciones(None) # Llamada inicial para poblar el combobox
        
        self.info_dist_generada = "" # Para guardar la info de la última generación
        # Botones
        btn_frame = ttk.Frame(frame, padding=(0, 10))
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        # --- CORRECCIÓN: Añadir todos los botones al mismo frame para que estén en la misma fila ---
        ttk.Button(btn_frame, text="🎲 Generar", command=self.generar_variables, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="📊 Visualizar", command=self.visualizar_datos, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="🧹 Limpiar", command=self.limpiar_generacion, style='Secondary.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="💾 Exportar Reporte", command=self.exportar_reporte_completo, style='Secondary.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="📋 Copiar Reporte", command=self.copiar_reporte_completo, style='Secondary.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="💾 Exportar Datos", command=self.exportar_solo_datos, style='Secondary.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="📋 Copiar Datos", command=self.copiar_solo_datos, style='Secondary.TButton').pack(side='left', padx=5)

        # Área de resultados
        result_frame = ttk.LabelFrame(frame, text="Resultados y Datos Generados", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Scrollbar para resultados
        scroll_y = ttk.Scrollbar(result_frame)
        scroll_y.pack(side='right', fill='y')
        
        scroll_x = ttk.Scrollbar(result_frame, orient='horizontal')
        scroll_x.pack(side='bottom', fill='x')
        
        self.resultado_text = tk.Text(result_frame, height=20, wrap='none',
                                     yscrollcommand=scroll_y.set, 
                                     xscrollcommand=scroll_x.set,
                                     font=self.font_code,
                                     bg=self.frame_bg_color, fg=self.text_color)
        self.resultado_text.pack(fill='both', expand=True)
        scroll_y.config(command=self.resultado_text.yview)
        scroll_x.config(command=self.resultado_text.xview)

        # --- MEJORA: Definir etiquetas de estilo para el texto de resultados ---
        self.resultado_text.tag_configure('title', font=self.font_title, foreground=self.accent_color, justify='center', spacing3=10)
        self.resultado_text.tag_configure('header', font=self.font_header, foreground=self.text_color, spacing3=5, background='#e2e8f0', lmargin1=5, lmargin2=5)
        self.resultado_text.tag_configure('label', font=self.font_bold, foreground=self.subtext_color)
        self.resultado_text.tag_configure('value', font=self.font_main, foreground=self.text_color)
        self.resultado_text.tag_configure('data', font=self.font_code, foreground=self.text_color)
        self.resultado_text.tag_configure('footer', font=('Segoe UI', 8, 'italic'), foreground=self.subtext_color, justify='center', spacing1=10)
        # Configurar tabulaciones para alinear estadísticas
        tabs = (180, 360) # Puntos de tabulación para la segunda columna
        self.resultado_text.config(tabs=tabs)
        
        self.datos_generados = None


    def actualizar_lista_distribuciones(self, event):
        """Actualiza el combobox de distribuciones basado en el tipo seleccionado."""
        tipo_seleccionado = self.dist_tipo_var.get()
        self.dist_combo['values'] = self.distribuciones[tipo_seleccionado]
        self.dist_combo.current(0) # Seleccionar el primer elemento de la nueva lista
        # Disparar la actualización de parámetros para la nueva selección
        self.actualizar_parametros_dist(None)
        
    def actualizar_parametros_dist(self, event):
        """Actualiza los campos de parámetros según la distribución seleccionada"""
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        self.param_entries = {}
        dist = self.dist_var.get()

        # --- REFACTORIZACIÓN: Usar un diccionario para la configuración ---
        dist_params_config = {
            'Bernoulli': [
                {'name': 'p', 'label': 'Probabilidad (p):', 'default': '0.5'}
            ],
            'Binomial': [
                {'name': 'n', 'label': 'Número de ensayos (n):', 'default': '10'},
                {'name': 'p', 'label': 'Probabilidad (p):', 'default': '0.5'}
            ],
            'Geometrica': [
                {'name': 'p', 'label': 'Probabilidad (p):', 'default': '0.5'}
            ],
            'Poisson': [
                {'name': 'lambda', 'label': 'Lambda (λ):', 'default': '5'}
            ],
            'Uniforme': [
                {'name': 'a', 'label': 'Límite inferior (a):', 'default': '0'},
                {'name': 'b', 'label': 'Límite superior (b):', 'default': '1'}
            ],
            'Exponencial': [
                {'name': 'lambda', 'label': 'Lambda (λ):', 'default': '1'}
            ],
            'Normal': [
                {'name': 'media', 'label': 'Media (μ):', 'default': '0'},
                {'name': 'std', 'label': 'Desviación estándar (σ):', 'default': '1'}
            ]
        }

        # Crear widgets dinámicamente desde la configuración
        if dist in dist_params_config:
            for i, param_info in enumerate(dist_params_config[dist]):
                name = param_info['name']
                label = param_info['label']
                default = param_info['default']

                ttk.Label(self.params_frame, text=label).grid(row=i, column=0, sticky='w', pady=5, padx=5)
                self.param_entries[name] = tk.StringVar(value=default)
                ttk.Entry(self.params_frame, textvariable=self.param_entries[name], width=25).grid(row=i, column=1, pady=5, padx=5)
    
    def generar_variables(self):
        """Genera variables aleatorias según los parámetros"""
        try:
            # Obtener semilla
            semilla_str = self.semilla_var.get().strip()
            semilla = int(semilla_str) if semilla_str else None
            
            # Crear generador
            generador = GeneradorAleatorio(semilla)
            
            # Obtener número de muestras
            n = int(self.n_muestras_var.get())
            
            if n <= 0 or n > 100000:
                raise ValueError("El número de muestras debe estar entre 1 y 100000")
            
            # --- OPTIMIZACIÓN: Usar un diccionario para la lógica de generación ---
            dist = self.dist_var.get()
            params = {key: var.get() for key, var in self.param_entries.items()}

            dist_logic = {
                'Bernoulli': lambda: (generador.bernoulli(float(params['p']), n), f"Bernoulli(p={params['p']})"),
                'Binomial': lambda: (generador.binomial(int(params['n']), float(params['p']), n), f"Binomial(n={params['n']}, p={params['p']})"),
                'Geometrica': lambda: (generador.geometrica(float(params['p']), n), f"Geometrica(p={params['p']})"),
                'Poisson': lambda: (generador.poisson(float(params['lambda']), n), f"Poisson(λ={params['lambda']})"),
                'Uniforme': lambda: (generador.uniforme(float(params['a']), float(params['b']), n), f"Uniforme(a={params['a']}, b={params['b']})"),
                'Exponencial': lambda: (generador.exponencial(float(params['lambda']), n), f"Exponencial(λ={params['lambda']})"),
                'Normal': lambda: (generador.normal(float(params['media']), float(params['std']), n), f"Normal(μ={params['media']}, σ={params['std']})")
            }

            if dist in dist_logic:
                self.datos_generados, info_dist = dist_logic[dist]()
                self.info_dist_generada = info_dist # Guardar para el reporte
                info = f"Distribución: {info_dist}"
            else:
                raise ValueError(f"Distribución '{dist}' no reconocida.")

            # --- MEJORA: Mostrar resultados con formato y estilos ---
            self.resultado_text.delete(1.0, tk.END)

            self.mostrar_reporte_generacion()
            
            messagebox.showinfo("Éxito", f"✓ {n} variables aleatorias generadas correctamente")
            
        except ValueError as e:
            messagebox.showerror("Error de Validación", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar variables: {str(e)}")
    
    def mostrar_reporte_generacion(self):
        """Muestra el reporte completo de la última generación de datos."""
        if self.datos_generados is None:
            messagebox.showwarning("Advertencia", "No hay datos para reportar. Genere variables primero.")
            return

        self.resultado_text.delete(1.0, tk.END)
        self.resultado_text.insert(tk.END, "Reporte de Generación de Variables Aleatorias\n", 'title')

        # Sección de Configuración
        self.resultado_text.insert(tk.END, "\n Configuración de la Simulación \n", 'header')
        # Asumimos que el generador y la info se guardaron en self
        generador = GeneradorAleatorio(int(self.semilla_var.get()) if self.semilla_var.get() else None)
        self.resultado_text.insert(tk.END, "Semilla utilizada:\t", 'label'); self.resultado_text.insert(tk.END, f"{generador.semilla}\n", 'value')
        self.resultado_text.insert(tk.END, "Distribución:\t", 'label'); self.resultado_text.insert(tk.END, f"{self.info_dist_generada}\n", 'value')
        self.resultado_text.insert(tk.END, "Muestras generadas:\t", 'label'); self.resultado_text.insert(tk.END, f"{len(self.datos_generados)}\n", 'value')

        # Sección de Estadísticas Descriptivas
        self.resultado_text.insert(tk.END, "\n Estadísticas Descriptivas \n", 'header')
        stats_data = {
            "Media:": np.mean(self.datos_generados),
            "Mediana:": np.median(self.datos_generados),
            "Desv. Estándar:": np.std(self.datos_generados),
            "Varianza:": np.var(self.datos_generados),
            "Mínimo:": np.min(self.datos_generados),
            "Máximo:": np.max(self.datos_generados),
            "Percentil 25 (Q1):": np.percentile(self.datos_generados, 25),
            "Percentil 75 (Q3):": np.percentile(self.datos_generados, 75)
        }
        
        keys = list(stats_data.keys())
        for i in range(0, len(keys), 2):
            key1 = keys[i]
            self.resultado_text.insert(tk.END, f"{key1}\t", 'label'); self.resultado_text.insert(tk.END, f"{stats_data[key1]:.6f}", 'value')
            if i + 1 < len(keys):
                key2 = keys[i+1]
                self.resultado_text.insert(tk.END, f"\t{key2}\t", 'label'); self.resultado_text.insert(tk.END, f"{stats_data[key2]:.6f}", 'value')
            self.resultado_text.insert(tk.END, "\n")

        # Sección de Datos Generados
        self.resultado_text.insert(tk.END, "\n Datos Generados \n", 'header')
        
        datos_por_fila = 10
        for i in range(0, len(self.datos_generados), datos_por_fila):
            fila = self.datos_generados[i:i+datos_por_fila]
            fila_str = "  ".join([f"{val:10.6f}" for val in fila])
            self.resultado_text.insert(tk.END, f"{fila_str}\n", 'data')
        
        self.resultado_text.insert(tk.END, "\n--- Fin del Reporte ---\n", 'footer')

    def copiar_reporte_completo(self):
        """Copia todo el contenido del reporte (cuadro de texto) al portapapeles"""
        if self.datos_generados is None:
            messagebox.showwarning("Advertencia", "No hay datos para copiar")
            return
        
        try:
            contenido = self.resultado_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(contenido)
            messagebox.showinfo("Éxito", "✓ El reporte completo ha sido copiado al portapapeles.", parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Error al copiar: {str(e)}")
    
    def copiar_solo_datos(self):
        """Copia únicamente los datos generados al portapapeles, uno por línea."""
        if self.datos_generados is None:
            messagebox.showwarning("Advertencia", "No hay datos para copiar. Genere variables primero.")
            return
        
        try:
            # Formatear los datos con un número por línea
            datos_str = "\n".join(map(str, self.datos_generados))
            self.root.clipboard_clear()
            self.root.clipboard_append(datos_str)
            messagebox.showinfo("Éxito", f"✓ {len(self.datos_generados)} datos han sido copiados al portapapeles (uno por línea).", parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Error al copiar los datos: {str(e)}")


    def exportar_reporte_completo(self):
        """Exporta TODO el contenido del reporte a un archivo de texto"""
        if self.datos_generados is None:
            messagebox.showwarning("Advertencia", "No hay datos para exportar")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")],
                title="Guardar Reporte Completo"
            )
            
            if filename:
                contenido = self.resultado_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(contenido)
                
                messagebox.showinfo("Éxito", f"✓ Reporte completo exportado a:\n{filename}", parent=self.root)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar datos: {str(e)}")

    def exportar_solo_datos(self):
        """Exporta únicamente los datos generados a un archivo de texto, uno por línea."""
        if self.datos_generados is None:
            messagebox.showwarning("Advertencia", "No hay datos para exportar. Genere variables primero.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")],
                title="Exportar solo los datos generados"
            )
            if filename:
                datos_str = "\n".join(map(str, self.datos_generados))
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(datos_str)
                messagebox.showinfo("Éxito", f"✓ {len(self.datos_generados)} datos exportados a:\n{filename}", parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar los datos: {str(e)}")
    
    def visualizar_datos(self):
        """Visualiza los datos generados en un histograma"""
        if self.datos_generados is None:
            messagebox.showwarning("Advertencia", "No hay datos para visualizar")
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.patch.set_facecolor(self.bg_color) # Fondo consistente

            # Histograma
            ax1.hist(self.datos_generados, bins=30, density=True, alpha=0.7, 
                    color=self.accent_color, edgecolor='black')
            ax1.set_title(f'Histograma - {self.dist_var.get()}', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Valor', fontsize=12)
            ax1.set_ylabel('Densidad', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # QQ-Plot
            stats.probplot(self.datos_generados, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normal)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            self.mostrar_grafico_con_opcion_guardar(fig, "Visualización de Datos Generados")
            # plt.close(fig) ya no es estrictamente necesario si la ventana se destruye
        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar: {str(e)}")

    def limpiar_generacion(self):
        """Limpia el área de resultados y los datos generados en la pestaña de generación."""
        self.resultado_text.delete(1.0, tk.END)
        self.datos_generados = None
        self.info_dist_generada = ""
        messagebox.showinfo("Limpieza", "✓ El área de resultados y los datos han sido limpiados.", parent=self.root)

    def mostrar_grafico_con_opcion_guardar(self, fig, window_title="Visualización"):
        """Muestra un gráfico de matplotlib en una nueva ventana de Tkinter con un botón para guardar."""
        try:
            plot_window = tk.Toplevel(self.root)
            plot_window.title(window_title)
            plot_window.geometry("900x700")
            plot_window.configure(bg=self.bg_color)

            main_frame = ttk.Frame(plot_window)
            main_frame.pack(fill='both', expand=True, padx=5, pady=5)

            canvas = FigureCanvasTkAgg(fig, master=main_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

            # --- MEJORA: Añadir la barra de herramientas de Matplotlib ---
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, main_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

            # Botón de guardar explícito (opcional, ya que la barra de herramientas lo tiene)
            def guardar_grafico():
                filepath = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("SVG Vector Image", "*.svg"), ("All Files", "*.*")]
                )
                if filepath:
                    # Usar bbox_inches='tight' para que no se corten los títulos/etiquetas
                    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
                    messagebox.showinfo("Éxito", f"Gráfico guardado en:\n{filepath}", parent=plot_window)

            # --- Frame para los botones inferiores ---
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(side='bottom', pady=5, fill='x')

            # Botón para cerrar la ventana
            ttk.Button(button_frame, text="💾 Guardar Gráfico", command=guardar_grafico).pack(side='left', expand=True, padx=5)
            ttk.Button(button_frame, text="Cerrar Ventana", command=plot_window.destroy).pack(side='left', expand=True, padx=5)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo mostrar la ventana del gráfico: {str(e)}")
    
    def crear_pestaña_bondad_ajuste(self):
        """Crea la pestaña de pruebas de bondad de ajuste"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Pruebas de Bondad de Ajuste")
        
        # Frame de entrada de datos
        input_frame = ttk.LabelFrame(frame, text="Entrada de Datos", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(input_frame, text="📁 Cargar desde archivo", command=self.cargar_datos_archivo, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(input_frame, text="✏️ Ingresar manualmente", command=self.ingresar_datos_manual, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(input_frame, text="🔄 Usar datos generados", command=self.usar_datos_generados, style='Accent.TButton').pack(side='left', padx=5)
        
        # Frame de configuración de prueba
        config_frame = ttk.LabelFrame(frame, text="Configuración de la Prueba", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(config_frame, text="Distribución a probar:").grid(row=0, column=0, sticky='w', pady=5, padx=5)
        self.dist_prueba_var = tk.StringVar()
        dist_combo = ttk.Combobox(config_frame, textvariable=self.dist_prueba_var, width=23, state='readonly')
        dist_combo['values'] = ('Normal', 'Exponencial', 'Uniforme')
        dist_combo.grid(row=0, column=1, pady=5, padx=5)
        dist_combo.current(0)
        
        ttk.Label(config_frame, text="Método de prueba:").grid(row=1, column=0, sticky='w', pady=5, padx=5)
        self.metodo_prueba_var = tk.StringVar()
        metodo_combo = ttk.Combobox(config_frame, textvariable=self.metodo_prueba_var, width=23, state='readonly')
        metodo_combo['values'] = ('Chi-cuadrado', 'Kolmogorov-Smirnov')
        metodo_combo.grid(row=1, column=1, pady=5, padx=5)
        metodo_combo.current(0)
        
        ttk.Label(config_frame, text="Nivel de significancia (α):").grid(row=2, column=0, sticky='w', pady=5, padx=5)
        self.alfa_var = tk.StringVar(value="0.05")
        ttk.Entry(config_frame, textvariable=self.alfa_var, width=25).grid(row=2, column=1, pady=5, padx=5)
        
        # --- Frame para botones de acción ---
        action_frame = ttk.Frame(config_frame)
        action_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(action_frame, text="▶️ Realizar Prueba", command=self.realizar_prueba_bondad, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(action_frame, text="💾 Exportar Resultados", command=self.exportar_resultados_bondad, style='Secondary.TButton').pack(side='left', padx=5)
        self.btn_visualizar_bondad = ttk.Button(action_frame, text="📊 Visualizar", command=self.visualizar_resultados_bondad, style='Secondary.TButton', state='disabled')
        self.btn_visualizar_bondad.pack(side='left', padx=5)
        ttk.Button(action_frame, text="📋 Copiar Resultados", command=self.copiar_resultados_bondad, style='Secondary.TButton').pack(side='left', padx=5)
        ttk.Button(action_frame, text="🧹 Limpiar", command=self.limpiar_bondad_ajuste, style='Secondary.TButton').pack(side='left', padx=5)
        
        # Área de resultados
        result_frame = ttk.LabelFrame(frame, text="Resultados", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        scroll = ttk.Scrollbar(result_frame)
        scroll.pack(side='right', fill='y')
        
        self.resultado_bondad_text = tk.Text(result_frame, height=15, yscrollcommand=scroll.set,
                                             font=self.font_code, bg=self.frame_bg_color, fg=self.text_color)
        self.resultado_bondad_text.pack(fill='both', expand=True)
        scroll.config(command=self.resultado_bondad_text.yview)
        
        # --- MEJORA: Definir etiquetas de estilo para el texto de resultados ---
        self.resultado_bondad_text.tag_configure('title', font=self.font_title, foreground=self.accent_color, justify='center', spacing3=10)
        self.resultado_bondad_text.tag_configure('header', font=self.font_header, foreground=self.text_color, spacing3=5, background='#e2e8f0', lmargin1=5, lmargin2=5)
        self.resultado_bondad_text.tag_configure('label', font=self.font_bold, foreground=self.subtext_color)
        self.resultado_bondad_text.tag_configure('value', font=self.font_main, foreground=self.text_color)
        self.resultado_bondad_text.tag_configure('success', font=self.font_bold, foreground=self.success_color)
        self.resultado_bondad_text.tag_configure('error', font=self.font_bold, foreground=self.error_color)

        self.datos_prueba = None
        # --- MEJORA: Variables para guardar el último resultado y poder visualizarlo de nuevo ---
        self.ultimo_resultado_bondad = None
        self.ultimo_metodo_bondad = None
        self.ultima_dist_bondad = None
        self.ultimos_params_bondad = None

    def limpiar_bondad_ajuste(self):
        """Limpia el área de resultados y los datos de la pestaña de bondad de ajuste."""
        self.resultado_bondad_text.delete(1.0, tk.END)
        self.datos_prueba = None
        self.ultimo_resultado_bondad = None
        self.ultimo_metodo_bondad = None
        self.ultima_dist_bondad = None
        self.ultimos_params_bondad = None
        self.btn_visualizar_bondad.config(state='disabled')
        messagebox.showinfo("Limpieza", "✓ Los resultados y datos de la prueba han sido limpiados.", parent=self.root)

    
    def cargar_datos_archivo(self):
        """Carga datos desde un archivo"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
            )
            
            if filename:
                datos = []
                with open(filename, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('='):
                            try:
                                # Intentar extraer números de la línea
                                numeros = line.split()
                                for num in numeros:
                                    try:
                                        datos.append(float(num))
                                    except ValueError:
                                        pass
                            except ValueError:
                                pass
                
                if len(datos) == 0:
                    raise ValueError("No se encontraron datos numéricos en el archivo")
                
                self.datos_prueba = np.array(datos)
                messagebox.showinfo("Éxito", f"✓ Se cargaron {len(self.datos_prueba)} datos")
                self.btn_visualizar_bondad.config(state='disabled') # Deshabilitar al cargar nuevos datos
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar archivo: {str(e)}")
    
    def ingresar_datos_manual(self):
        """Permite ingresar datos manualmente"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Ingresar Datos Manualmente")
        dialog.geometry("500x400")
        dialog.configure(bg=self.bg_color)
        
        ttk.Label(dialog, text="Ingrese los datos separados por comas, espacios o líneas:").pack(pady=10)
        
        text = tk.Text(dialog, height=15, font=self.font_code)
        text.pack(fill='both', expand=True, padx=10, pady=5)
        
        def guardar_datos():
            try:
                contenido = text.get(1.0, tk.END).strip()
                # Separar por comas, espacios o líneas
                contenido = contenido.replace(',', ' ').replace('\n', ' ')
                datos = [float(x) for x in contenido.split() if x]
                
                if len(datos) == 0:
                    raise ValueError("No se ingresaron datos válidos")
                
                self.datos_prueba = np.array(datos)
                messagebox.showinfo("Éxito", f"✓ Se ingresaron {len(self.datos_prueba)} datos")
                self.btn_visualizar_bondad.config(state='disabled') # Deshabilitar al cargar nuevos datos
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Error al procesar datos: {str(e)}")
        
        ttk.Button(dialog, text="Guardar", command=guardar_datos).pack(pady=10)
    
    def usar_datos_generados(self):
        """Usa los datos generados en la pestaña anterior"""
        if self.datos_generados is None:
            messagebox.showwarning("Advertencia", "No hay datos generados. Use la primera pestaña para generar datos.")
            return
        
        self.datos_prueba = self.datos_generados.copy()
        messagebox.showinfo("Éxito", f"✓ Se usarán {len(self.datos_prueba)} datos generados")
        self.btn_visualizar_bondad.config(state='disabled') # Deshabilitar al cargar nuevos datos

    def copiar_resultados_bondad(self):
        """Copia el contenido del cuadro de resultados de bondad de ajuste al portapapeles."""
        contenido = self.resultado_bondad_text.get(1.0, tk.END).strip()
        if not contenido:
            messagebox.showwarning("Advertencia", "No hay resultados para copiar. Realice una prueba primero.")
            return
        
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(contenido)
            messagebox.showinfo("Éxito", "✓ Los resultados de la prueba han sido copiados al portapapeles.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al copiar los resultados: {str(e)}")

    def exportar_resultados_bondad(self):
        """Exporta los resultados de la prueba de bondad de ajuste a un archivo de texto."""
        contenido = self.resultado_bondad_text.get(1.0, tk.END).strip()
        if not contenido:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar. Realice una prueba primero.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")],
                title="Guardar Resultados de la Prueba"
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(contenido)
                messagebox.showinfo("Éxito", f"✓ Resultados exportados a:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar los resultados: {str(e)}")
    
    def visualizar_resultados_bondad(self):
        """Vuelve a mostrar la visualización de la última prueba de bondad de ajuste realizada."""
        if self.ultimo_resultado_bondad is None:
            messagebox.showwarning("Advertencia", "No hay resultados de prueba para visualizar. Realice una prueba primero.")
            return

        try:
            if self.ultimo_metodo_bondad == 'Chi-cuadrado':
                self.visualizar_comparacion(self.ultimo_resultado_bondad, self.ultima_dist_bondad, self.ultimos_params_bondad)
            elif self.ultimo_metodo_bondad == 'Kolmogorov-Smirnov':
                self.visualizar_comparacion_ks(self.ultima_dist_bondad, self.ultimos_params_bondad)
        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar los resultados: {str(e)}")

    def _guardar_ultimo_resultado_bondad(self, resultado, metodo, distribucion, params):
        """Guarda los detalles de la última prueba para poder visualizarlos de nuevo."""
        self.ultimo_resultado_bondad, self.ultimo_metodo_bondad, self.ultima_dist_bondad, self.ultimos_params_bondad = resultado, metodo, distribucion, params
        self.btn_visualizar_bondad.config(state='normal') # Habilitar el botón de visualización

    def realizar_prueba_bondad(self):
        """Realiza la prueba de bondad de ajuste"""
        if self.datos_prueba is None:
            messagebox.showwarning("Advertencia", "Primero debe cargar o generar datos")
            return
        
        try:
            distribucion = self.dist_prueba_var.get().lower()
            metodo = self.metodo_prueba_var.get()
            alfa = float(self.alfa_var.get())
            
            if not 0 < alfa < 1:
                raise ValueError("El nivel de significancia debe estar entre 0 y 1")
            
            # Estimar parámetros
            if distribucion == 'normal':
                params = (np.mean(self.datos_prueba), np.std(self.datos_prueba))
                param_str = f"μ={params[0]:.4f}, σ={params[1]:.4f}"                    
            elif distribucion == 'exponencial':
                params = (1/np.mean(self.datos_prueba),)
                param_str = f"λ={params[0]:.4f}"
            elif distribucion == 'uniforme':
                params = (np.min(self.datos_prueba), np.max(self.datos_prueba))
                param_str = f"a={params[0]:.4f}, b={params[1]:.4f}"
            
            # Realizar prueba
            prueba = PruebasBondadAjuste()
            
            if metodo == 'Chi-cuadrado':
                resultado = prueba.chi_cuadrado(self.datos_prueba, distribucion, params)
                
                # --- MEJORA: Mostrar resultados con formato ---
                self.resultado_bondad_text.delete(1.0, tk.END)
                self.resultado_bondad_text.insert(tk.END, "Prueba Chi-Cuadrado de Bondad de Ajuste\n", 'title')

                self.resultado_bondad_text.insert(tk.END, "\n Configuración de la Prueba \n", 'header')
                self.resultado_bondad_text.insert(tk.END, "Distribución a probar:\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{distribucion.capitalize()}\n", 'value')
                self.resultado_bondad_text.insert(tk.END, "Parámetros estimados:\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{param_str}\n", 'value')
                self.resultado_bondad_text.insert(tk.END, "Tamaño de muestra:\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{len(self.datos_prueba)}\n", 'value')

                self.resultado_bondad_text.insert(tk.END, "\n Resultados \n", 'header')
                self.resultado_bondad_text.insert(tk.END, "Estadístico χ²:\t\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{resultado['estadistico']:.6f}\n", 'value')
                self.resultado_bondad_text.insert(tk.END, "Grados de libertad:\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{resultado['grados_libertad']}\n", 'value')
                self.resultado_bondad_text.insert(tk.END, "Valor p:\t\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{resultado['p_valor']:.6f}\n", 'value')
                self.resultado_bondad_text.insert(tk.END, "Nivel de significancia α:\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{alfa}\n", 'value')

                self.resultado_bondad_text.insert(tk.END, "\n Conclusión \n", 'header')
                
                if resultado['p_valor'] > alfa:
                    self.resultado_bondad_text.insert(tk.END, "✓ NO SE RECHAZA H₀\n", 'success')
                    self.resultado_bondad_text.insert(tk.END, f"La evidencia sugiere que los datos se ajustan a una distribución {distribucion.capitalize()}.\n", 'value')
                    self.resultado_bondad_text.insert(tk.END, f"(p-valor = {resultado['p_valor']:.4f} > α = {alfa})\n", 'value')
                else:
                    self.resultado_bondad_text.insert(tk.END, "✗ SE RECHAZA H₀\n", 'error')
                    self.resultado_bondad_text.insert(tk.END, f"La evidencia sugiere que los datos NO se ajustan a una distribución {distribucion.capitalize()}.\n", 'value')
                    self.resultado_bondad_text.insert(tk.END, f"(p-valor = {resultado['p_valor']:.4f} ≤ α = {alfa})\n", 'value')

                # Guardar resultado para poder visualizarlo de nuevo y habilitar botón
                self._guardar_ultimo_resultado_bondad(resultado, metodo, distribucion, params)

                # Visualizar comparación para Chi-Cuadrado
                self.visualizar_comparacion(resultado, distribucion, params)
                
            elif metodo == 'Kolmogorov-Smirnov':
                resultado = prueba.kolmogorov_smirnov(self.datos_prueba, distribucion, params)
                
                # --- MEJORA: Mostrar resultados con formato ---
                self.resultado_bondad_text.delete(1.0, tk.END)
                self.resultado_bondad_text.insert(tk.END, "Prueba Kolmogorov-Smirnov de Bondad de Ajuste\n", 'title')

                self.resultado_bondad_text.insert(tk.END, "\n Configuración de la Prueba \n", 'header')
                self.resultado_bondad_text.insert(tk.END, "Distribución a probar:\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{distribucion.capitalize()}\n", 'value')
                self.resultado_bondad_text.insert(tk.END, "Parámetros estimados:\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{param_str}\n", 'value')
                self.resultado_bondad_text.insert(tk.END, "Tamaño de muestra:\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{len(self.datos_prueba)}\n", 'value')

                self.resultado_bondad_text.insert(tk.END, "\n Resultados \n", 'header')
                self.resultado_bondad_text.insert(tk.END, "Estadístico D:\t\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{resultado['estadistico']:.6f}\n", 'value')
                self.resultado_bondad_text.insert(tk.END, "Valor p:\t\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{resultado['p_valor']:.6f}\n", 'value')
                self.resultado_bondad_text.insert(tk.END, "Nivel de significancia α:\t", 'label'); self.resultado_bondad_text.insert(tk.END, f"{alfa}\n", 'value')

                self.resultado_bondad_text.insert(tk.END, "\n Conclusión \n", 'header')
                
                if resultado['p_valor'] > alfa:
                    self.resultado_bondad_text.insert(tk.END, "✓ NO SE RECHAZA H₀\n", 'success')
                    self.resultado_bondad_text.insert(tk.END, f"La evidencia sugiere que los datos se ajustan a una distribución {distribucion.capitalize()}.\n", 'value')
                    self.resultado_bondad_text.insert(tk.END, f"(p-valor = {resultado['p_valor']:.4f} > α = {alfa})\n", 'value')
                else:
                    self.resultado_bondad_text.insert(tk.END, "✗ SE RECHAZA H₀\n", 'error')
                    self.resultado_bondad_text.insert(tk.END, f"La evidencia sugiere que los datos NO se ajustan a una distribución {distribucion.capitalize()}.\n", 'value')
                    self.resultado_bondad_text.insert(tk.END, f"(p-valor = {resultado['p_valor']:.4f} ≤ α = {alfa})\n", 'value')
                
                # Guardar resultado para poder visualizarlo de nuevo y habilitar botón
                self._guardar_ultimo_resultado_bondad(resultado, metodo, distribucion, params)

                # Visualizar comparación para K-S
                self.visualizar_comparacion_ks(distribucion, params)
        
        except ValueError as e:
            messagebox.showerror("Error de Validación", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Error en la prueba: {str(e)}")
    
    def visualizar_comparacion(self, resultado, distribucion, params):
        """Visualiza la comparación entre datos observados y esperados"""
        try:
            # --- CORRECCIÓN: Mejorar el layout para evitar que los gráficos se vean alterados/superpuestos ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                           gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})
            fig.patch.set_facecolor(self.bg_color)

            # Gráfico 1: Histograma con curva teórica
            ax1.hist(self.datos_prueba, bins=30, density=True, alpha=0.7,
                    color=self.accent_color, edgecolor='black', label='Datos observados')
            
            x = np.linspace(min(self.datos_prueba), max(self.datos_prueba), 100)
            if distribucion == 'normal':
                y = stats.norm.pdf(x, params[0], params[1])
            elif distribucion == 'exponencial':
                y = stats.expon.pdf(x, scale=1/params[0])
            elif distribucion == 'uniforme':
                y = stats.uniform.pdf(x, params[0], params[1]-params[0])
            
            ax1.plot(x, y, 'r-', linewidth=3, label=f'Distribución {distribucion}')
            ax1.set_title('Comparación: Observado vs Teórico', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Valor', fontsize=12)
            ax1.set_ylabel('Densidad', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Frecuencias observadas vs esperadas
            bins = resultado['bins']
            widths = bins[1:] - bins[:-1]
            ax2.bar(bins[:-1], resultado['observado'], width=widths, align='edge', alpha=0.7,
                   color=self.accent_color, label='Observado', edgecolor='black')
            ax2.plot((bins[:-1] + bins[1:]) / 2, resultado['esperado'], 'r-o', linewidth=3,
                    label='Esperado', markersize=8)
            ax2.set_title('Frecuencias: Observadas vs Esperadas', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Valor', fontsize=12)
            ax2.set_ylabel('Frecuencia', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            self.mostrar_grafico_con_opcion_guardar(fig, "Comparación para Prueba Chi-Cuadrado")
            plt.close(fig) # Cerrar la figura para liberar memoria

        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar: {str(e)}")
    
    def visualizar_comparacion_ks(self, distribucion, params):
        """Visualiza la comparación de CDF para la prueba K-S"""
        try:            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                           gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})
            fig.patch.set_facecolor(self.bg_color)

            # --- Gráfico 1: Comparación de CDF (Función de Distribución Acumulada) ---
            datos_sorted = np.sort(self.datos_prueba)
            ecdf = np.arange(1, len(datos_sorted) + 1) / len(datos_sorted)
            ax1.step(datos_sorted, ecdf, label='CDF Empírica (Datos)', where='post', color=self.accent_color, linewidth=2)
            
            x_lims = (min(self.datos_prueba), max(self.datos_prueba))
            x_teorico = np.linspace(x_lims[0], x_lims[1], 200)
            
            if distribucion == 'normal':
                dist_obj = stats.norm(params[0], params[1])
                dist_label = f'CDF Normal (μ={params[0]:.2f}, σ={params[1]:.2f})'
            elif distribucion == 'exponencial':
                dist_obj = stats.expon(scale=1/params[0])
                dist_label = f'CDF Exponencial (λ={params[0]:.2f})'
            elif distribucion == 'uniforme':
                dist_obj = stats.uniform(loc=params[0], scale=params[1]-params[0])
                dist_label = f'CDF Uniforme (a={params[0]:.2f}, b={params[1]:.2f})'
            
            y_teorico = dist_obj.cdf(x_teorico)
            ax1.plot(x_teorico, y_teorico, 'r--', linewidth=2.5, label=dist_label)
            
            ax1.set_title('Comparación de CDF', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Valor', fontsize=12)
            ax1.set_ylabel('Probabilidad Acumulada', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            ax1.set_xlim(x_lims)

            # --- Gráfico 2: Histograma vs PDF Teórica ---
            ax2.hist(self.datos_prueba, bins=30, density=True, alpha=0.7, 
                    color=self.success_color, edgecolor='black', label='Datos observados')
            y_pdf = dist_obj.pdf(x_teorico)
            ax2.plot(x_teorico, y_pdf, 'r-', linewidth=3, label=f'PDF Teórica {distribucion.capitalize()}')
            ax2.set_title('Comparación: Histograma vs PDF', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Valor', fontsize=12)
            ax2.set_ylabel('Densidad', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)

            self.mostrar_grafico_con_opcion_guardar(fig, "Comparación para Prueba K-S")
            plt.close(fig) # Cerrar la figura para liberar memoria
        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar la comparación K-S: {str(e)}")

    def visualizar_linea_tiempo_colas(self, event_log, titulo, num_servidores=1):
        """Crea un gráfico de Gantt para visualizar la simulación de colas."""
        if not event_log:
            return

        # Limitar el número de clientes a mostrar para no saturar el gráfico
        max_clientes_a_mostrar = 50
        mostrar_anotaciones = len(event_log) <= 15
        log_a_mostrar = event_log[:max_clientes_a_mostrar]

        fig, ax = plt.subplots(figsize=(15, 8))
        fig.patch.set_facecolor(self.bg_color)

        # --- MEJORA: Añadir subtítulo si los datos están truncados ---
        if len(event_log) > max_clientes_a_mostrar:
            fig.suptitle(titulo, fontsize=16, fontweight='bold')
            ax.set_title(f"(Mostrando los primeros {max_clientes_a_mostrar} de {len(event_log)} clientes)", fontsize=10)
        else:
            ax.set_title(titulo, fontsize=16, fontweight='bold')

        if num_servidores == 1:
            # Para M/M/1, el eje Y son los clientes
            y_labels = [f"Cliente {e['cliente_id']}" for e in log_a_mostrar]
            y_ticks = range(len(log_a_mostrar))
            ax.set_yticks(y_ticks, labels=reversed(y_labels))  # Invertir para que el Cliente 1 esté arriba

            for i, evento in enumerate(reversed(log_a_mostrar)):
                y_pos = i
                # Dibujar barra de espera (rojo)
                if evento['espera'] > 1e-6: # Solo dibujar si la espera es significativa
                    ax.broken_barh([(evento['llegada'], evento['espera'])], (y_pos - 0.4, 0.8), facecolors=self.error_color, edgecolor='black', linewidth=0.5)
                # Dibujar barra de servicio (verde)
                ax.broken_barh([(evento['inicio_servicio'], evento['duracion_servicio'])], (y_pos - 0.4, 0.8), facecolors=self.success_color, edgecolor='black', linewidth=0.5)

                # --- MEJORA: Añadir anotaciones de texto si no son demasiados clientes ---
                if mostrar_anotaciones:
                    # Marcador de llegada
                    ax.plot(evento['llegada'], y_pos, 'o', color='#3b82f6', markersize=6)
                    ax.text(evento['llegada'], y_pos + 0.5, f"L: {evento['llegada']:.2f}", ha='center', va='bottom', fontsize=8)
                    # Anotación de fin de servicio
                    ax.text(evento['fin_servicio'], y_pos, f"{evento['fin_servicio']:.2f}", ha='left', va='center', fontsize=8, color=self.subtext_color)

        # --- MEJORA: Añadir leyenda ---
        import matplotlib.patches as mpatches
        legend_patches = [
            mpatches.Patch(color=self.error_color, label='Tiempo de Espera'),
            mpatches.Patch(color=self.success_color, label='Tiempo de Servicio'),
        ]
        if mostrar_anotaciones:
            legend_patches.append(plt.Line2D([0], [0], marker='o', color='w', label='Llegada del Cliente', markerfacecolor='#3b82f6', markersize=8))
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')

        ax.set_xlabel("Línea de Tiempo", fontsize=12)
        ax.set_ylabel("Cliente", fontsize=12)
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        plt.subplots_adjust(right=0.85) # Ajustar para dar espacio a la leyenda
        
        self.mostrar_grafico_con_opcion_guardar(fig, titulo)
        plt.close(fig)

    def crear_pestaña_monte_carlo(self):
        """
        Crea la pestaña de la GUI para las simulaciones Monte Carlo.

        Esta pestaña permite al usuario seleccionar una simulación, configurar sus
        parámetros y ejecutarla, mostrando los resultados y gráficos correspondientes.
        """
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Método de Monte Carlo")
        
        # Selector de problema
        select_frame = ttk.LabelFrame(frame, text="Seleccionar Problema", padding=10)
        select_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(select_frame, text="Tipo de simulación:").grid(row=0, column=0, sticky='w', pady=5, padx=5)
        self.problema_var = tk.StringVar()
        problema_combo = ttk.Combobox(select_frame, textvariable=self.problema_var, width=35, state='readonly')
        problema_combo['values'] = ('Estimación de π (Pi)', 'Ruina del Jugador', 'Sistema de Colas M/M/1')
        problema_combo.grid(row=0, column=1, pady=5, padx=10)
        problema_combo.current(0)
        problema_combo.bind('<<ComboboxSelected>>', self.actualizar_parametros_mc)
        
        # Frame de parámetros
        self.mc_params_frame = ttk.LabelFrame(frame, text="Parámetros", padding=10)
        self.mc_params_frame.pack(fill='x', padx=10, pady=5)
        
        # --- MEJORA: Contenedor para los botones de acción ---
        self.mc_action_buttons_frame = ttk.Frame(frame)
        self.mc_action_buttons_frame.pack(pady=10)

        # Botón principal de simulación
        self.mc_run_button = ttk.Button(self.mc_action_buttons_frame, text="▶️ Ejecutar Simulación", command=self.ejecutar_monte_carlo, style='Accent.TButton')
        self.mc_run_button.pack(side='left', padx=5)
        self.mc_cola_event_log = None # Para guardar el log de eventos de la cola

        self.mc_param_entries = {}
        self.actualizar_parametros_mc(None)

        # --- MEJORA: Barra de progreso ---
        progress_frame = ttk.Frame(frame)
        # Este frame se empaquetará dinámicamente
        progress_frame.pack(fill='x', padx=10, pady=5)
        self.mc_progress_label = ttk.Label(progress_frame, text="Progreso:")
        self.mc_progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', length=300, mode='determinate')
        self.mc_progress_percent_label = ttk.Label(progress_frame, text="0%")
        # Inicialmente ocultos
        self.mc_progress_label.pack_forget()
        self.mc_progress_bar.pack_forget()
        self.mc_progress_percent_label.pack_forget()
        
        # Área de resultados
        result_frame = ttk.LabelFrame(frame, text="Resultados", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        scroll = ttk.Scrollbar(result_frame)
        scroll.pack(side='right', fill='y')
        
        self.resultado_mc_text = tk.Text(result_frame, height=15, yscrollcommand=scroll.set,
                                         font=self.font_code, bg=self.frame_bg_color, fg=self.text_color)
        self.resultado_mc_text.pack(fill='both', expand=True)
        scroll.config(command=self.resultado_mc_text.yview)

        # --- MEJORA: Definir etiquetas de estilo para el texto de resultados ---
        self.resultado_mc_text.tag_configure('title', font=self.font_title, foreground=self.accent_color, justify='center', spacing3=10)
        self.resultado_mc_text.tag_configure('header', font=self.font_header, foreground=self.text_color, spacing3=5, background='#e2e8f0', lmargin1=5, lmargin2=5)
        self.resultado_mc_text.tag_configure('label', font=self.font_bold, foreground=self.subtext_color)
        self.resultado_mc_text.tag_configure('value', font=self.font_main, foreground=self.text_color)
        self.resultado_mc_text.tag_configure('footer', font=('Segoe UI', 8, 'italic'), foreground=self.subtext_color, justify='center', spacing1=10)
    
    def actualizar_parametros_mc(self, event):
        """Actualiza los parámetros según el problema seleccionado"""
        # Limpiar frame
        for widget in self.mc_params_frame.winfo_children():
            widget.destroy()        

        # --- MEJORA: Limpiar botones extra (como el de la línea de tiempo) al cambiar de simulación ---
        # Destruir todos los widgets hijos excepto el primero (que es el botón de ejecutar)
        for widget in self.mc_action_buttons_frame.winfo_children()[1:]:
            widget.destroy()

        self.mc_cola_event_log = None
        
        self.mc_param_entries = {}
        problema = self.problema_var.get()

        # --- MEJORA: Crear un frame contenedor para parámetros y ejemplo ---
        container = ttk.Frame(self.mc_params_frame)
        container.pack(fill='x', expand=True)
        params_container = ttk.Frame(container)
        params_container.grid(row=0, column=0, sticky='ns')
        example_container = ttk.Frame(container)
        example_container.grid(row=0, column=1, sticky='ns', padx=(20, 0))
        container.grid_columnconfigure(1, weight=1)

        def crear_ejemplo(parent, titulo, texto):
            example_frame = ttk.LabelFrame(parent, text=f"💡 Ejemplo: {titulo}", padding=10)
            example_frame.pack(fill='both', expand=True)
            label = ttk.Label(example_frame, text=texto, wraplength=450, justify='left', font=self.font_main)
            label.pack(fill='both', expand=True)

        if problema == 'Estimación de π (Pi)':
            ttk.Label(params_container, text="Número de simulaciones:").grid(row=0, column=0, sticky='w', pady=5, padx=5)
            self.mc_param_entries['n_sim'] = tk.StringVar(value="10000")
            ttk.Entry(params_container, textvariable=self.mc_param_entries['n_sim'], width=25).grid(row=0, column=1, pady=5, padx=5)
            
        elif problema == 'Ruina del Jugador':
            ttk.Label(params_container, text="Capital inicial:").grid(row=0, column=0, sticky='w', pady=5, padx=5)
            self.mc_param_entries['capital'] = tk.StringVar(value="50")
            ttk.Entry(params_container, textvariable=self.mc_param_entries['capital'], width=25).grid(row=0, column=1, pady=5, padx=5)
            
            ttk.Label(params_container, text="Probabilidad de ganar:").grid(row=1, column=0, sticky='w', pady=5, padx=5)
            self.mc_param_entries['prob'] = tk.StringVar(value="0.48")
            ttk.Entry(params_container, textvariable=self.mc_param_entries['prob'], width=25).grid(row=1, column=1, pady=5, padx=5)
            
            ttk.Label(params_container, text="Apuesta por jugada:").grid(row=2, column=0, sticky='w', pady=5, padx=5)
            self.mc_param_entries['apuesta'] = tk.StringVar(value="1")
            ttk.Entry(params_container, textvariable=self.mc_param_entries['apuesta'], width=25).grid(row=2, column=1, pady=5, padx=5)
            
            ttk.Label(params_container, text="Objetivo (capital):").grid(row=3, column=0, sticky='w', pady=5, padx=5)
            self.mc_param_entries['objetivo'] = tk.StringVar(value="100")
            ttk.Entry(params_container, textvariable=self.mc_param_entries['objetivo'], width=25).grid(row=3, column=1, pady=5, padx=5)
            
            ttk.Label(params_container, text="Número de simulaciones:").grid(row=4, column=0, sticky='w', pady=5, padx=5)
            self.mc_param_entries['n_sim'] = tk.StringVar(value="1000")
            ttk.Entry(params_container, textvariable=self.mc_param_entries['n_sim'], width=25).grid(row=4, column=1, pady=5, padx=5)

            crear_ejemplo(example_container, "Apuestas en la Ruleta",
                          "Un jugador va a un casino con $50. Apuesta $1 en cada giro al 'rojo' en la ruleta (prob. de ganar ≈ 0.48). Su objetivo es llegar a $100 antes de quedarse sin dinero.\n\n¿Cuál es la probabilidad de que termine en la ruina (pierda todo su dinero)?")
            
        elif problema == 'Sistema de Colas M/M/1':
            ttk.Label(params_container, text="Tasa de llegada (λ):").grid(row=0, column=0, sticky='w', pady=5, padx=5)
            self.mc_param_entries['lambda_ll'] = tk.StringVar(value="2.0")
            ttk.Entry(params_container, textvariable=self.mc_param_entries['lambda_ll'], width=25).grid(row=0, column=1, pady=5, padx=5)
            
            ttk.Label(params_container, text="Tasa de servicio (μ):").grid(row=1, column=0, sticky='w', pady=5, padx=5)
            self.mc_param_entries['mu'] = tk.StringVar(value="3.0")
            ttk.Entry(params_container, textvariable=self.mc_param_entries['mu'], width=25).grid(row=1, column=1, pady=5, padx=5)
            
            ttk.Label(params_container, text="Tiempo de simulación:").grid(row=2, column=0, sticky='w', pady=5, padx=5)
            self.mc_param_entries['tiempo'] = tk.StringVar(value="100")
            ttk.Entry(params_container, textvariable=self.mc_param_entries['tiempo'], width=25).grid(row=2, column=1, pady=5, padx=5)

            crear_ejemplo(example_container, "Caja Rápida en Supermercado",
                          "A una caja rápida llegan en promedio 2 clientes por minuto (λ=2). El cajero puede atender a 3 clientes por minuto (μ=3).\n\n¿Cuál es el tiempo promedio que un cliente debe esperar en la fila antes de ser atendido?")

    def ejecutar_monte_carlo(self):
        """
        Prepara y lanza la simulación de Monte Carlo en un hilo secundario.

        Esto evita que la interfaz de usuario se congele durante cálculos intensivos.
        Muestra una barra de progreso y deshabilita el botón de ejecución.
        """
        if self.is_mc_running:
            messagebox.showwarning("Simulación en Curso", "Por favor, espere a que la simulación actual termine.")
            return

        self.is_mc_running = True
        self.mc_run_button.config(state="disabled")

        # Preparar la barra de progreso
        self.mc_progress_label.pack(side='left', padx=5)
        self.mc_progress_bar.pack(side='left', fill='x', expand=True, padx=5)
        self.mc_progress_percent_label.pack(side='left', padx=5)
        self.mc_progress_bar['value'] = 0
        self.mc_progress_percent_label['text'] = "0%"
        
        self.resultado_mc_text.delete(1.0, tk.END)
        self.resultado_mc_text.insert(tk.END, "⚙️  Iniciando simulación en segundo plano...\n")
        self.root.update_idletasks()

        # Crear y empezar el hilo
        thread = threading.Thread(target=self._worker_monte_carlo, daemon=True)
        thread.start()

    def _worker_monte_carlo(self):
        """
        Función de trabajo que se ejecuta en el hilo secundario.

        Realiza la simulación seleccionada, llama al `progress_callback` para
        actualizar la GUI y, al finalizar, programa la actualización final de
        resultados en el hilo principal usando `root.after()`.
        """
        try:
            problema = self.problema_var.get()

            # Función para actualizar la GUI de forma segura desde el hilo
            def progress_callback(paso, total):
                if total > 0:
                    porcentaje = (paso / total) * 100
                    self.root.after(0, self._update_mc_progress, porcentaje)

            if problema == 'Estimación de π (Pi)':
                n_sim = int(self.mc_param_entries['n_sim'].get())
                if n_sim <= 0: raise ValueError("El número de simulaciones debe ser positivo.")
                
                resultado = MonteCarlo.estimar_pi(n_sim, progress_callback=progress_callback)
                params = {'n_sim': n_sim}

            elif problema == 'Ruina del Jugador':
                capital = int(self.mc_param_entries['capital'].get())
                prob = float(self.mc_param_entries['prob'].get())
                apuesta = int(self.mc_param_entries['apuesta'].get())
                objetivo = int(self.mc_param_entries['objetivo'].get())
                n_sim = int(self.mc_param_entries['n_sim'].get())

                resultado = MonteCarlo.ruina_jugador(capital, prob, apuesta, objetivo, n_sim, progress_callback=progress_callback)
                params = {'capital': capital, 'prob': prob, 'apuesta': apuesta, 'objetivo': objetivo, 'n_sim': n_sim}

            elif problema == 'Sistema de Colas M/M/1':
                lambda_ll = float(self.mc_param_entries['lambda_ll'].get())
                mu = float(self.mc_param_entries['mu'].get())
                tiempo = float(self.mc_param_entries['tiempo'].get())

                if lambda_ll >= mu:
                    self.root.after(0, messagebox.showwarning, "Advertencia", "El sistema es inestable (λ >= μ). La cola crecerá indefinidamente.")
                
                resultado = MonteCarlo.cola_mm1(lambda_ll, mu, tiempo, progress_callback=progress_callback)
                params = {'lambda_ll': lambda_ll, 'mu': mu, 'tiempo': tiempo}
            # Cuando la simulación termina, actualiza la GUI final desde el hilo principal
            self.root.after(0, self._finalize_mc_simulation, resultado, problema, params)

        except ValueError as e:
            self.root.after(0, messagebox.showerror, "Error de Validación", str(e))
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Error al ejecutar la simulación: {str(e)}")
        finally:
            # Reactivar la UI, siempre desde el hilo principal
            self.root.after(0, self._reset_mc_ui)

    def _update_mc_progress(self, porcentaje):
        """Actualiza la barra de progreso (método seguro para hilos)."""
        self.mc_progress_bar['value'] = porcentaje
        self.mc_progress_percent_label['text'] = f"{porcentaje:.1f}%"

    def _reset_mc_ui(self):
        """Limpia la UI de progreso y reactiva los controles después de una simulación."""
        self.mc_progress_label.pack_forget()
        self.mc_progress_bar.pack_forget()
        self.mc_progress_percent_label.pack_forget()
        self.mc_run_button.config(state="normal")
        self.is_mc_running = False

    def _finalize_mc_simulation(self, resultado, problema, params):
        """Muestra los resultados y gráficos finales en la GUI una vez que la simulación ha terminado."""
        # --- MEJORA: Mostrar resultados con formato ---
        self.resultado_mc_text.delete(1.0, tk.END)
        self.resultado_mc_text.insert(tk.END, f"Simulación Monte Carlo: {problema}\n", 'title')

        if problema == 'Estimación de π (Pi)':
            n_sim = params['n_sim']
            self.resultado_mc_text.insert(tk.END, "\n Configuración \n", 'header')
            self.resultado_mc_text.insert(tk.END, f"Número de simulaciones:\t{n_sim}\n", 'label')

            self.resultado_mc_text.insert(tk.END, "\n Resultados \n", 'header')
            self.resultado_mc_text.insert(tk.END, f"Valor estimado de π:\t{resultado['pi_estimado']:.8f}\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Valor real de π:\t\t{math.pi:.8f}\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Puntos dentro:\t\t{resultado['dentro']}\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Puntos totales:\t\t{resultado['total']}\n", 'label')

            # --- CORRECCIÓN: El gráfico de convergencia podía mostrarse mal ---
            # Se crea una figura con un layout más flexible (GridSpec) para acomodar los dos gráficos.
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            fig.patch.set_facecolor(self.bg_color) # Fondo consistente

            ax2.plot(range(1, n_sim + 1), resultado['historial_pi'], label='Estimación de π')
            ax2.axhline(y=math.pi, color='r', linestyle='--', label=f'Valor real de π ({math.pi:.6f})')
            ax2.set_title('Convergencia de la Estimación', fontsize=14, fontweight='bold')
            if n_sim > 1000: ax2.set_xscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            ax1.scatter(resultado['puntos_x'], resultado['puntos_y'], c=resultado['colores'], s=1, alpha=0.5)
            ax1.add_artist(plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2))
            ax1.set_title(f'Estimación de π con {n_sim} puntos', fontsize=14, fontweight='bold')
            ax1.set_aspect('equal', adjustable='box')
            ax1.grid(True, alpha=0.3)

            self.mostrar_grafico_con_opcion_guardar(fig, "Simulación - Estimación de Pi")
            plt.close(fig) # Liberar memoria

        elif problema == 'Ruina del Jugador':
            self.resultado_mc_text.insert(tk.END, "\n Configuración \n", 'header')
            self.resultado_mc_text.insert(tk.END, f"Capital inicial:\t\t{params['capital']}\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Prob. de ganar (p):\t{params['prob']}\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Apuesta:\t\t\t{params['apuesta']}\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Objetivo:\t\t\t{params['objetivo']}\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Número de simulaciones:\t{params['n_sim']}\n", 'label')
            
            self.resultado_mc_text.insert(tk.END, "\n Resultados \n", 'header')
            self.resultado_mc_text.insert(tk.END, f"Prob. de ruina (simulada):\t{resultado['prob_ruina']:.4f} ({resultado['prob_ruina']*100:.2f}%)\n", 'label')
            
            p = params['prob']
            # --- CORRECCIÓN: La fórmula teórica de la ruina del jugador debe usar unidades de apuesta ---
            # La fórmula requiere que el capital y el objetivo estén en "unidades de apuesta".
            apuesta_val = params['apuesta']
            if apuesta_val <= 0: raise ValueError("La apuesta debe ser positiva.")
            capital_unidades = params['capital'] / apuesta_val
            objetivo_unidades = params['objetivo'] / apuesta_val

            if p != 0.5:
                q = 1 - p
                ratio = q / p
                # Usar math.pow para mayor precisión con flotantes
                prob_ruina_teorica = (math.pow(ratio, objetivo_unidades) - math.pow(ratio, capital_unidades)) / (math.pow(ratio, objetivo_unidades) - 1)
            else:
                prob_ruina_teorica = 1 - (capital_unidades / objetivo_unidades)

            self.resultado_mc_text.insert(tk.END, f"Prob. de ruina (teórica):\t{prob_ruina_teorica:.4f} ({prob_ruina_teorica*100:.2f}%)\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Prob. de éxito (simulada):\t{resultado['prob_exito']:.4f} ({resultado['prob_exito']*100:.2f}%)\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Duración promedio:\t\t{resultado['duracion_promedio']:.2f} jugadas\n", 'label')

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(self.bg_color) # Fondo consistente

            ax.hist(resultado['duraciones'], bins=50, color=self.success_color, edgecolor='black', alpha=0.7)
            ax.set_title('Distribución de la Duración del Juego', fontsize=14, fontweight='bold')
            ax.set_xlabel('Número de Jugadas', fontsize=12)
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.grid(True, alpha=0.3)
            self.mostrar_grafico_con_opcion_guardar(fig, "Simulación - Ruina del Jugador")
            plt.close(fig) # Liberar memoria

        elif problema == 'Sistema de Colas M/M/1':
            self.resultado_mc_text.insert(tk.END, "\n Configuración \n", 'header')
            self.resultado_mc_text.insert(tk.END, f"Tasa de llegada (λ):\t{params['lambda_ll']}\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Tasa de servicio (μ):\t{params['mu']}\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Tiempo de simulación:\t{params['tiempo']}\n", 'label')

            self.resultado_mc_text.insert(tk.END, "\n Resultados Simulados \n", 'header')
            self.resultado_mc_text.insert(tk.END, f"Clientes atendidos:\t\t{resultado['clientes_atendidos']}\n", 'label')
            self.resultado_mc_text.insert(tk.END, f"Tiempo de espera prom. (Wq):\t{resultado['tiempo_espera_promedio']:.4f}\n", 'label')
            
            if params['lambda_ll'] < params['mu']:
                rho = params['lambda_ll'] / params['mu']
                Wq_teorico = (rho / (params['mu'] - params['lambda_ll']))
                self.resultado_mc_text.insert(tk.END, "\n Resultados Teóricos (M/M/1) \n", 'header')
                self.resultado_mc_text.insert(tk.END, f"Factor de utilización (ρ):\t{rho:.4f}\n", 'label')
                self.resultado_mc_text.insert(tk.END, f"Tiempo de espera prom. (Wq):\t{Wq_teorico:.4f}\n", 'label')

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(self.bg_color) # Fondo consistente

            ax.hist(resultado['tiempos_espera'], bins=50, color='#f97316', edgecolor='black', alpha=0.7) # Naranja
            ax.set_title('Distribución de Tiempos de Espera en Cola', fontsize=14, fontweight='bold')
            ax.set_xlabel('Tiempo de Espera', fontsize=12)
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.grid(True, alpha=0.3)
            self.mostrar_grafico_con_opcion_guardar(fig, "Simulación de Colas - Tiempos de Espera")

            # --- MEJORA: Crear botón para mostrar la línea de tiempo bajo demanda ---
            self.mc_cola_event_log = resultado['event_log']
            # Limpiar botones extra anteriores (por si se ejecuta varias veces)
            for widget in self.mc_action_buttons_frame.winfo_children()[1:]:
                widget.destroy()
            
            btn_timeline = ttk.Button(self.mc_action_buttons_frame, text="📊 Ver Línea de Tiempo",
                                      command=lambda: self.visualizar_linea_tiempo_colas(self.mc_cola_event_log, titulo="Análisis de la Línea de Tiempo (M/M/1)"),
                                      style='Secondary.TButton')
            btn_timeline.pack(side='left', padx=5)

            plt.close(fig) # Liberar memoria

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionSimulacion(root)
    root.mainloop()