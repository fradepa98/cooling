# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:35:48 2021

@author: Francesco De Palo
"""

import pandas as pd
import numpy as np
import psychro as psy

# constants
c = 1e3         # J/kg K, air specific heat
l = 2496e3      # J/kg, latent heat

# to be used in self.m_ls / least_squares
m_max = 100     # ks/s, max dry air mass flow rate
θs_0 = 5        # °C, initial guess for saturation temperature

"""
GENERAL MODEL AND LEGEND

<=5=================================5=========<==================
  mo      ||                        m                          ||
          5 (m-mo) =======1=======                             ||
          ||       ||  (1-β)m   ||                             ||
θo,φo=0=>[MX1]==1==||          [MX2]==3(=4)=F==>======[TZ]==5==||
 mo             |  ||           ||                     //      |
                |  ===1=[CC]==2===                     sl      |
                |        \\   βm                       ||      |
                |         sl                          [BL]<-mi |
                |         |                                    |
                |         |                                    |
                |         |                                    |
                |--------------------------------[K]-----------|<-φI
                          -----------------------[K]-----------|<-θI

LEGEND:
Inputs:
θo, φo      outdoor temperature & humidity ratio
θIsp, φIsp  indoor temperature & humidity ratio set points
mi          infiltration mass flow rate
Qsa, Qla    auxiliary sensible and latent loads [kW]

Parameters:
m           mass flow rate of dry air (depending on the problem)
α           fraction of fresh air
β           by-pass factir od cooling coil (depending on the problem)
UA          overall heat transfer coefficient

Elements (13 equations):
MX1         mixing box (2 equations)
CC          cooling coil (4 equations)
MX2         mixing box (2 equations)
TZ          thermal zone (2 equations)
BL          building (2 equations)
Kθ          indoor temperature controller (1 equation)
F           fan

Outputs (13 unknowns):
0, 1, 2, 3   temperature and humidity ratio (8 unknowns)
Qt, Qs, Ql  total, sensible and latent heat of CC (3 unknowns)
Qs, Ql      sensible and latent heat of TZ (2 unknowns)

REMARK: In order to obtain the most general model possible, the reheating part
is introduced, although switched off (Kw=0)
"""


class MxCcTzBl:
    """
    **HVAC composition**:
        mixing, cooling, reaheating, thermal zone of building, recycling
    """

    def __init__(self, parameters, inputs):
        m, mo, β, Kθ, Kw = parameters
        θo, φo, θIsp, φIsp, mi, UA, Qsa, Qla = inputs

        self.design = np.array([m, mo, β, Kθ, Kw,       # parameters
                                θo, φo, θIsp, φIsp,     # inputs air out, in
                                mi, UA, Qsa, Qla])      # --"--  building
        self.actual = np.array([m, mo, β, Kθ, Kw,
                                θo, φo, θIsp, φIsp,
                                mi, UA, Qsa, Qla])

    def lin_model(self, θs0):

        m, mo, β, Kθ, Kw, θo, φo, θIsp, φIsp, mi, UA, Qsa, Qla = self.actual
        wo = psy.w(θo, φo)      # hum. out

        A = np.zeros((16, 16))  # coefficents of unknowns
        b = np.zeros(16)        # vector of inputs
        # MX1
        A[0, 0], A[0, 8], b[0] = m * c, -(m - mo) * c, mo * c * θo
        A[1, 1], A[1, 9], b[1] = m * l, -(m - mo) * l, mo * l * wo
        # CC
        A[2, 0], A[2, 2], A[2, 11], b[2] = (1 - β) * m * c, -(1 - β) * m * c,\
            1, 0
        A[3, 1], A[3, 3], A[3, 12], b[3] = (1 - β) * m * l, -(1 - β) * m * l,\
            1, 0
        A[4, 2], A[4, 3], b[4] = psy.wsp(θs0), -1,\
            psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        A[5, 10], A[5, 11], A[5, 12], b[5] = -1, 1, 1, 0
        # MX2
        A[6, 0], A[6, 2], A[6, 4], b[6] = β * m * c, (1 - β) * m * c,\
            - m * c, 0
        A[7, 1], A[7, 3], A[7, 5], b[7] = β * m * l, (1 - β) * m * l,\
            - m * l, 0
        # HC
        A[8, 4], A[8, 6], A[8, 13], b[8] = m * c, -m * c, 1, 0
        A[9, 5], A[9, 7], b[9] = m * l, -m * l, 0
        # TZ
        A[10, 6], A[10, 8], A[10, 14], b[10] = m * c, -m * c, 1, 0
        A[11, 7], A[11, 9], A[11, 15], b[11] = m * l, -m * l, 1, 0
        # BL
        A[12, 8], A[12, 14], b[12] = (UA + mi * c), 1, (UA + mi * c) * θo + Qsa
        A[13, 9], A[13, 15], b[13] = mi * l, 1, mi * l * wo + Qla
        # Kθ indoor temperature controller
        A[14, 8], A[14, 10], b[14] = Kθ, 1, Kθ * θIsp
        # Kw indoor humidity ratio controller
        A[15, 9], A[15, 13], b[15] = Kw, 1, Kw * psy.w(θIsp, φIsp)
        x = np.linalg.solve(A, b)
        return x

    def solve_lin(self, θs0):
        """
        Finds saturation point on saturation curve ws = f(θs).
            Solves iterativelly *lin_model(θs0)*:
            θs -> θs0 until ws = psy(θs, 1).

        Parameters
        ----------
        θs0     initial guess saturation temperature
        """

        Δ_ws = 10e-3  # kg/kg, initial difference to start the iterations
        while Δ_ws > 0.01e-3:
            x = self.lin_model(θs0)
            Δ_ws = abs(psy.w(x[2], 1) - x[3])   # psy.w(θs, 1) = ws
            θs0 = x[2]                          # actualize θs0
        return x

    def m_ls(self, value, sp):
        """
        Mass flow rate m controls supply temperature θS or indoor humidity wI.
            Finds m which solves value = sp, i.e. minimizes ε = value - sp.
            Uses *scipy.optimize.least_squares* to solve the non-linear system.

        Parameters
        ----------
        value   string: 'θS' od 'wI' type of controlled variable
        sp      float: value of setpoint

        Calls
        -----
        *ε(m)*  gives (value - sp) to be minimized for m

        Returns (16 unknowns)
        ---------------------
        x           given by *self.lin_model(self, θs0)*
        """
        from scipy.optimize import least_squares

        def ε(m):
            """
            Gives difference ε = (values - sp) function of m
                ε  calculated by self.solve_lin(ts0)
                m   bounds=(0, m_max); m_max hard coded (global variable)

            Parameters
            ----------
            m : mass flow rate of dry air
            """
            self.actual[0] = m
            x = self.solve_lin(θs_0)
            if value == 'θS':
                θS = x[6]       # supply air
                return abs(sp - θS)
            elif value == 'φI':
                wI = x[9]       # indoor air
                return abs(sp - wI)
            else:
                print('ERROR in ε(m): value not in {"θS", "wI"}')

        m0 = self.actual[0]     # initial guess
        if value == 'φI':
            self.actual[4] = 0
            sp = psy.w(self.actual[7], sp)
        # gives m for min(θSsp - θS); θs_0 is the initial guess of θs
        res = least_squares(ε, m0, bounds=(0, m_max))

        if res.cost < 0.1e-3:
            m = float(res.x)
            # print(f'm = {m: 5.3f} kg/s')
        else:
            print('RecAirVAV: No solution for m')

        self.actual[0] = m

        x = self.solve_lin(θs_0)
        θo = 32
        φo = 0.5
        self.psy_chart(x, θo, φo)

        print(f'Apparatus dew point temperature is: {x[2]: .3f} °C')
        print(f'Total load on the cooling coil is: {x[10]: .2f} W')
        print(f'Air mass flow rate is: {m: .3f} kg/s')

        return x

    def β_ls(self, value, sp):
        """
        Bypass β controls supply temperature θS or indoor humidity wI.
            Finds β which solves value = sp, i.e. minimizes ε = value - sp.
            Uses *scipy.optimize.least_squares* to solve the non-linear system.

        Parameters
        ----------
        value   string: 'θS' od 'wI' type of controlled variable
        sp      float: value of setpoint

        """
        from scipy.optimize import least_squares

        def ε(β):
            """
            Gives difference ε = (values - sp) function of β
                ε  calculated by self.solve_lin(ts0)
                β   bounds=(0, 1)
            """
            self.actual[2] = β
            x = self.solve_lin(θs_0)
            if value == 'θS':
                θS = x[6]       # supply air
                return abs(sp - θS)
            elif value == 'φI':
                wI = x[9]       # indoor air
                return abs(sp - wI)
            else:
                print('ERROR in ε(β): value not in {"θS", "wI"}')

        β0 = self.actual[2]     # initial guess
        β0 = 0.1
        if value == 'φI':
            self.actual[4] = 0
            sp = psy.w(self.actual[7], sp)
        # gives m for min(θSsp - θS); θs_0 is the initial guess of θs
        res = least_squares(ε, β0, bounds=(0, 1))

        if res.cost < 1e-5:
            β = float(res.x)
            # print(f'm = {m: 5.3f} kg/s')
        else:
            print('RecAirVBP: No solution for β')

        self.actual[2] = β
        x = self.solve_lin(θs_0)
        return x

    def psy_chart(self, x, θo, φo):
        """
        Plot results on psychrometric chart.

        Parameters
        ----------
        x : θM, wM, θs, ws, θC, wC, θS, wS, θI, wI,
            QtCC, QsCC, QlCC, QsHC, QsTZ, QlTZ
                    results of self.solve_lin or self.m_ls
        θo, φo      outdoor point

        Returns
        -------
        None.

        """
        # Processes on psychrometric chart
        wo = psy.w(θo, φo)
        # Points: O, s, S, I
        θ = np.append(θo, x[0:10:2])
        w = np.append(wo, x[1:10:2])
        # Points       O   s   S   I     Elements
        A = np.array([[-1, 1, 0, 0, 0, 1],      # MX1
                      [0, -1, 1, 0, 0, 0],      # CC
                      [0, 0, -1, 1, -1, 0],     # MX2
                      [0, 0, 0, -1, 1, 0],      # HC
                      [0, 0, 0, 0, -1, 1]])     # TZ
        psy.chartA(θ, w, A)

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)         # kg/kg -> g/kg
        P = pd.concat([θ, w], axis=1)   # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format})
        print()
        print(output)

        Q = pd.Series(x[10:], index=['QtCC', 'QsCC', 'QlCC', 'QsHC',
                                     'QsTZ', 'QlTZ'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')
        return None

    def VBP_wd(self, value='θS', sp=18, θo=32, φo=0.5, θIsp=24, φIsp=0.5,
               mi=1.35, UA=675, Qsa=34_000, Qla=4_000):
        """
        Variable by-pass (VBP) to be used in Jupyter with widgets

        Parameters
        ----------
        value       {"θS", "wI"}' type of value controlled
        sp          set point for the controlled value
        θo, φo, θIsp, φIsp, mi, UA, Qsa, Qla
                    given by widgets in Jupyter Lab

        """

        # Design values
        self.actual[5:] = θo, φo, θIsp, φIsp, mi, UA, Qsa, Qla

        x = self.β_ls(value, sp)
        print('m = {m: .3f} kg/s, mo = {mo: .3f} kg/s, β = {β: .3f}'.format(
            m=self.actual[0], mo=self.actual[1], β=self.actual[2]))
        self.psy_chart(x, θo, φo)
        
        print(f'Apparatus dew point temperature is: {x[2]: .3f} °C')
        print(f'Total load on the cooling coil is: {x[10]: .2f} W')
        print(f'Air mass flow rate is: {self.actual[0]: .3f} kg/s')

        return x
