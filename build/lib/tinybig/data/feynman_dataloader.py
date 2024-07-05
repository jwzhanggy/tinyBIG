# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from tinybig.data.function_dataloader import function_dataloader


# these 100 Feynman Equations are downloaded from https://space.mit.edu/home/tegmark/aifeynman.html
# credit belongs to the Feynman Symbolic Regression Database.
# format: "Filename,Number,Output,Formula,# variables,v1_name,v1_low,v1_high,v2_name,v2_low,v2_high,v3_name,v3_low,v3_high,v4_name,v4_low,v4_high,v5_name,v5_low,v5_high,v6_name,v6_low,v6_high,v7_name,v7_low,v7_high,v8_name,v8_low,v8_high,v9_name,v9_low,v9_high,v10_name,v10_low,v10_high"
Feynman_Equations = (
    "I.6.2a,1,f,exp(-theta**2/2)/sqrt(2*pi),1,theta,1,3,,,,,,,,,,,,,,,,,,,,,,,,,,,",
    "I.6.2,2,f,exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma),2,sigma,1,3,theta,1,3,,,,,,,,,,,,,,,,,,,,,,,,",
    "I.6.2b,3,f,exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma),3,sigma,1,3,theta,1,3,theta1,1,3,,,,,,,,,,,,,,,,,,,,,",
    "I.8.14,4,d,sqrt((x2-x1)**2+(y2-y1)**2),4,x1,1,5,x2,1,5,y1,1,5,y2,1,5,,,,,,,,,,,,,,,,,,",
    "I.9.18,5,F,G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2),9,m1,1,2,m2,1,2,G,1,2,x1,3,4,x2,1,2,y1,3,4,y2,1,2,z1,3,4,z2,1,2,,,",
    "I.10.7,6,m,m_0/sqrt(1-v**2/c**2),3,m_0,1,5,v,1,2,c,3,10,,,,,,,,,,,,,,,,,,,,,",
    "I.11.19,7,A,x1*y1+x2*y2+x3*y3,6,x1,1,5,x2,1,5,x3,1,5,y1,1,5,y2,1,5,y3,1,5,,,,,,,,,,,,",
    "I.12.1,8,F,mu*Nn,2,mu,1,5,Nn,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "I.13.4,9,K,1/2*m*(v**2+u**2+w**2),4,m,1,5,v,1,5,u,1,5,w,1,5,,,,,,,,,,,,,,,,,,",
    "I.12.2,10,F,q1*q2*r/(4*pi*epsilon*r**3),4,q1,1,5,q2,1,5,epsilon,1,5,r,1,5,,,,,,,,,,,,,,,,,,",
    "I.12.4,11,Ef,q1*r/(4*pi*epsilon*r**3),3,q1,1,5,epsilon,1,5,r,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.12.5,12,F,q2*Ef,2,q2,1,5,Ef,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "I.12.11,13,F,q*(Ef+B*v*sin(theta)),5,q,1,5,Ef,1,5,B,1,5,v,1,5,theta,1,5,,,,,,,,,,,,,,,",
    "I.13.12,14,U,G*m1*m2*(1/r2-1/r1),5,m1,1,5,m2,1,5,r1,1,5,r2,1,5,G,1,5,,,,,,,,,,,,,,,",
    "I.14.3,15,U,m*g*z,3,m,1,5,g,1,5,z,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.14.4,16,U,1/2*k_spring*x**2,2,k_spring,1,5,x,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "I.15.3x,17,x1,(x-u*t)/sqrt(1-u**2/c**2),4,x,5,10,u,1,2,c,3,20,t,1,2,,,,,,,,,,,,,,,,,,",
    "I.15.3t,18,t1,(t-u*x/c**2)/sqrt(1-u**2/c**2),4,x,1,5,c,3,10,u,1,2,t,1,5,,,,,,,,,,,,,,,,,,",
    "I.15.1,19,p,m_0*v/sqrt(1-v**2/c**2),3,m_0,1,5,v,1,2,c,3,10,,,,,,,,,,,,,,,,,,,,,",
    "I.16.6,20,v1,(u+v)/(1+u*v/c**2),3,c,1,5,v,1,5,u,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.18.4,21,r,(m1*r1+m2*r2)/(m1+m2),4,m1,1,5,m2,1,5,r1,1,5,r2,1,5,,,,,,,,,,,,,,,,,,",
    "I.18.12,22,tau,r*F*sin(theta),2,r,1,5,F,1,5,theta,0,5,,,,,,,,,,,,,,,,,,,,,",
    "I.18.14,23,L,m*r*v*sin(theta),3,m,1,5,r,1,5,v,1,5,theta,1,5,,,,,,,,,,,,,,,,,,",
    "I.24.6,24,E_n,1/2*m*(omega**2+omega_0**2)*1/2*x**2,4,m,1,3,omega,1,3,omega_0,1,3,x,1,3,,,,,,,,,,,,,,,,,,",
    "I.25.13,25,Volt,q/C,2,q,1,5,C,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "I.26.2,26,theta1,arcsin(n*sin(theta2)),2,n,0,1,theta2,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "I.27.6,27,foc,1/(1/d1+n/d2),3,d1,1,5,d2,1,5,n,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.29.4,28,k,omega/c,2,omega,1,10,c,1,10,,,,,,,,,,,,,,,,,,,,,,,,",
    "I.29.16,29,x,sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2)),4,x1,1,5,x2,1,5,theta1,1,5,theta2,1,5,,,,,,,,,,,,,,,,,,",
    "I.30.3,30,Int,Int_0*sin(n*theta/2)**2/sin(theta/2)**2,3,Int_0,1,5,theta,1,5,n,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.30.5,31,theta,arcsin(lambd/(n*d)),3,lambd,1,2,d,2,5,n,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.32.5,32,Pwr,q**2*a**2/(6*pi*epsilon*c**3),4,q,1,5,a,1,5,epsilon,1,5,c,1,5,,,,,,,,,,,,,,,,,,",
    "I.32.17,33,Pwr,(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2),6,epsilon,1,2,c,1,2,Ef,1,2,r,1,2,omega,1,2,omega_0,3,5,,,,,,,,,,,,",
    "I.34.8,34,omega,q*v*B/p,4,q,1,5,v,1,5,B,1,5,p,1,5,,,,,,,,,,,,,,,,,,",
    "I.34.1,35,omega,omega_0/(1-v/c),3,c,3,10,v,1,2,omega_0,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.34.14,36,omega,(1+v/c)/sqrt(1-v**2/c**2)*omega_0,3,c,3,10,v,1,2,omega_0,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.34.27,37,E_n,(h/(2*pi))*omega,2,omega,1,5,h,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "I.37.4,38,Int,I1+I2+2*sqrt(I1*I2)*cos(delta),3,I1,1,5,I2,1,5,delta,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.38.12,39,r,4*pi*epsilon*(h/(2*pi))**2/(m*q**2),3,m,1,5,q,1,5,h,1,5,epsilon,1,5,,,,,,,,,,,,,,,,,,",
    "I.39.1,40,E_n,3/2*pr*V,2,pr,1,5,V,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "I.39.11,41,E_n,1/(gamma-1)*pr*V,3,gamma,2,5,pr,1,5,V,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.39.22,42,pr,n*kb*T/V,4,n,1,5,T,1,5,V,1,5,kb,1,5,,,,,,,,,,,,,,,,,,",
    "I.40.1,43,n,n_0*exp(-m*g*x/(kb*T)),6,n_0,1,5,m,1,5,x,1,5,T,1,5,g,1,5,kb,1,5,,,,,,,,,,,,",
    "I.41.16,44,L_rad,h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1)),5,omega,1,5,T,1,5,h,1,5,kb,1,5,c,1,5,,,,,,,,,,,,,,,",
    "I.43.16,45,v,mu_drift*q*Volt/d,4,mu_drift,1,5,q,1,5,Volt,1,5,d,1,5,,,,,,,,,,,,,,,,,,",
    "I.43.31,46,D,mob*kb*T,3,mob,1,5,T,1,5,kb,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.43.43,47,kappa,1/(gamma-1)*kb*v/A,4,gamma,2,5,kb,1,5,A,1,5,v,1,5,,,,,,,,,,,,,,,,,,",
    "I.44.4,48,E_n,n*kb*T*ln(V2/V1),5,n,1,5,kb,1,5,T,1,5,V1,1,5,V2,1,5,,,,,,,,,,,,,,,",
    "I.47.23,49,c,sqrt(gamma*pr/rho),3,gamma,1,5,pr,1,5,rho,1,5,,,,,,,,,,,,,,,,,,,,,",
    "I.48.2,50,E_n,m*c**2/sqrt(1-v**2/c**2),3,m,1,5,v,1,2,c,3,10,,,,,,,,,,,,,,,,,,,,,",
    "I.50.26,51,x,x1*(cos(omega*t)+alpha*cos(omega*t)**2),4,x1,1,3,omega,1,3,t,1,3,alpha,1,3,,,,,,,,,,,,,,,,,,",
    "II.2.42,52,Pwr,kappa*(T2-T1)*A/d,5,kappa,1,5,T1,1,5,T2,1,5,A,1,5,d,1,5,,,,,,,,,,,,,,,",
    "II.3.24,53,flux,Pwr/(4*pi*r**2),2,Pwr,1,5,r,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "II.4.23,54,Volt,q/(4*pi*epsilon*r),3,q,1,5,epsilon,1,5,r,1,5,,,,,,,,,,,,,,,,,,,,,",
    "II.6.11,55,Volt,1/(4*pi*epsilon)*p_d*cos(theta)/r**2,4,epsilon,1,3,p_d,1,3,theta,1,3,r,1,3,,,,,,,,,,,,,,,,,,",
    "II.6.15a,56,Ef,p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2),6,epsilon,1,3,p_d,1,3,r,1,3,x,1,3,y,1,3,z,1,3,,,,,,,,,,,,",
    "II.6.15b,57,Ef,p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3,4,epsilon,1,3,p_d,1,3,theta,1,3,r,1,3,,,,,,,,,,,,,,,,,,",
    "II.8.7,58,E_n,3/5*q**2/(4*pi*epsilon*d),3,q,1,5,epsilon,1,5,d,1,5,,,,,,,,,,,,,,,,,,,,,",
    "II.8.31,59,E_den,epsilon*Ef**2/2,2,epsilon,1,5,Ef,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "II.10.9,60,Ef,sigma_den/epsilon*1/(1+chi),3,sigma_den,1,5,epsilon,1,5,chi,1,5,,,,,,,,,,,,,,,,,,,,,",
    "II.11.3,61,x,q*Ef/(m*(omega_0**2-omega**2)),5,q,1,3,Ef,1,3,m,1,3,omega_0,3,5,omega,1,2,,,,,,,,,,,,,,,",
    "II.11.17,62,n,n_0*(1+p_d*Ef*cos(theta)/(kb*T)),6,n_0,1,3,kb,1,3,T,1,3,theta,1,3,p_d,1,3,Ef,1,3,,,,,,,,,,,,",
    "II.11.20,63,Pol,n_rho*p_d**2*Ef/(3*kb*T),5,n_rho,1,5,p_d,1,5,Ef,1,5,kb,1,5,T,1,5,,,,,,,,,,,,,,,",
    "II.11.27,64,Pol,n*alpha/(1-(n*alpha/3))*epsilon*Ef,4,n,0,1,alpha,0,1,epsilon,1,2,Ef,1,2,,,,,,,,,,,,,,,,,,",
    "II.11.28,65,theta,1+n*alpha/(1-(n*alpha/3)),2,n,0,1,alpha,0,1,,,,,,,,,,,,,,,,,,,,,,,,",
    "II.13.17,66,B,1/(4*pi*epsilon*c**2)*2*I/r,4,epsilon,1,5,c,1,5,I,1,5,r,1,5,,,,,,,,,,,,,,,,,,",
    "II.13.23,67,rho_c,rho_c_0/sqrt(1-v**2/c**2),3,rho_c_0,1,5,v,1,2,c,3,10,,,,,,,,,,,,,,,,,,,,,",
    "II.13.34,68,j,rho_c_0*v/sqrt(1-v**2/c**2),3,rho_c_0,1,5,v,1,2,c,3,10,,,,,,,,,,,,,,,,,,,,,",
    "II.15.4,69,E_n,-mom*B*cos(theta),3,mom,1,5,B,1,5,theta,1,5,,,,,,,,,,,,,,,,,,,,,",
    "II.15.5,70,E_n,-p_d*Ef*cos(theta),3,p_d,1,5,Ef,1,5,theta,1,5,,,,,,,,,,,,,,,,,,,,,",
    "II.21.32,71,Volt,q/(4*pi*epsilon*r*(1-v/c)),5,q,1,5,epsilon,1,5,r,1,5,v,1,2,c,3,10,,,,,,,,,,,,,,,",
    "II.24.17,72,k,sqrt(omega**2/c**2-pi**2/d**2),3,omega,4,6,c,1,2,d,2,4,,,,,,,,,,,,,,,,,,,,,",
    "II.27.16,73,flux,epsilon*c*Ef**2,3,epsilon,1,5,c,1,5,Ef,1,5,,,,,,,,,,,,,,,,,,,,,",
    "II.27.18,74,E_den,epsilon*Ef**2,2,epsilon,1,5,Ef,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "II.34.2a,75,I,q*v/(2*pi*r),3,q,1,5,v,1,5,r,1,5,,,,,,,,,,,,,,,,,,,,,",
    "II.34.2,76,mom,q*v*r/2,3,q,1,5,v,1,5,r,1,5,,,,,,,,,,,,,,,,,,,,,",
    "II.34.11,77,omega,g_*q*B/(2*m),4,g_,1,5,q,1,5,B,1,5,m,1,5,,,,,,,,,,,,,,,,,,",
    "II.34.29a,78,mom,q*h/(4*pi*m),3,q,1,5,h,1,5,m,1,5,,,,,,,,,,,,,,,,,,,,,",
    "II.34.29b,79,E_n,g_*mom*B*Jz/(h/(2*pi)),5,g_,1,5,h,1,5,Jz,1,5,mom,1,5,B,1,5,,,,,,,,,,,,,,,",
    "II.35.18,80,n,n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T))),5,n_0,1,3,kb,1,3,T,1,3,mom,1,3,B,1,3,,,,,,,,,,,,,,,",
    "II.35.21,81,M,n_rho*mom*tanh(mom*B/(kb*T)),5,n_rho,1,5,mom,1,5,B,1,5,kb,1,5,T,1,5,,,,,,,,,,,,,,,",
    "II.36.38,82,f,mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M,8,mom,1,3,H,1,3,kb,1,3,T,1,3,alpha,1,3,epsilon,1,3,c,1,3,M,1,3,,,,,,",
    "II.37.1,83,E_n,mom*(1+chi)*B,6,mom,1,5,B,1,5,chi,1,5,,,,,,,,,,,,,,,,,,,,,",
    "II.38.3,84,F,Y*A*x/d,4,Y,1,5,A,1,5,d,1,5,x,1,5,,,,,,,,,,,,,,,,,,",
    "II.38.14,85,mu_S,Y/(2*(1+sigma)),2,Y,1,5,sigma,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "III.4.32,86,n,1/(exp((h/(2*pi))*omega/(kb*T))-1),4,h,1,5,omega,1,5,kb,1,5,T,1,5,,,,,,,,,,,,,,,,,,",
    "III.4.33,87,E_n,(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1),4,h,1,5,omega,1,5,kb,1,5,T,1,5,,,,,,,,,,,,,,,,,,",
    "III.7.38,88,omega,2*mom*B/(h/(2*pi)),3,mom,1,5,B,1,5,h,1,5,,,,,,,,,,,,,,,,,,,,,",
    "III.8.54,89,prob,sin(E_n*t/(h/(2*pi)))**2,3,E_n,1,2,t,1,2,h,1,4,,,,,,,,,,,,,,,,,,,,,",
    "III.9.52,90,prob,(p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2,6,p_d,1,3,Ef,1,3,t,1,3,h,1,3,omega,1,5,omega_0,1,5,,,,,,,,,,,,",
    "III.10.19,91,E_n,mom*sqrt(Bx**2+By**2+Bz**2),4,mom,1,5,Bx,1,5,By,1,5,Bz,1,5,,,,,,,,,,,,,,,,,,",
    "III.12.43,92,L,n*(h/(2*pi)),2,n,1,5,h,1,5,,,,,,,,,,,,,,,,,,,,,,,,",
    "III.13.18,93,v,2*E_n*d**2*k/(h/(2*pi)),4,E_n,1,5,d,1,5,k,1,5,h,1,5,,,,,,,,,,,,,,,,,,",
    "III.14.14,94,I,I_0*(exp(q*Volt/(kb*T))-1),5,I_0,1,5,q,1,2,Volt,1,2,kb,1,2,T,1,2,,,,,,,,,,,,,,,",
    "III.15.12,95,E_n,2*U*(1-cos(k*d)),3,U,1,5,k,1,5,d,1,5,,,,,,,,,,,,,,,,,,,,,",
    "III.15.14,96,m,(h/(2*pi))**2/(2*E_n*d**2),3,h,1,5,E_n,1,5,d,1,5,,,,,,,,,,,,,,,,,,,,,",
    "III.15.27,97,k,2*pi*alpha/(n*d),3,alpha,1,5,n,1,5,d,1,5,,,,,,,,,,,,,,,,,,,,,",
    "III.17.37,98,f,Beta*(1+alpha*cos(theta)),3,Beta,1,5,alpha,1,5,theta,1,5,,,,,,,,,,,,,,,,,,,,,",
    "III.19.51,99,E_n,-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2),4,m,1,5,q,1,5,h,1,5,n,1,5,epsilon,1,5,,,,,,,,,,,,,,,",
    "III.21.20,100,j,-rho_c_0*q*A_vec/m,4,rho_c_0,1,5,q,1,5,A_vec,1,5,m,1,5,,,,,,,,,,,,,,,,,,"
)

Dimensionless_Feynman_Equations = (
    "I.6.2,0,f,exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma),2,sigma,0,1,theta,0,1",
    "I.6.2b,1,f,exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma),3,sigma,0,1,theta,0,1,theta1,0,1",
    "I.9.18,2,F,a/((b-1)**2+(c-d)**2+(e-f)**2),6,a,0,1,b,0,1,c,0,1,d,0,1,e,0,1,f,0,1",
    "I.12.11,3,F,1+a*sin(theta),2,a,0,1,theta,0,1",
    "I.13.12,4,U,a*(1/b-1),2,a,0,1,b,0,1",
    "I.15.3x,5,x1,(1-a)/sqrt(1-b**2),2,a,0,1,b,0,1",
    "I.16.6,6,v1,(a+b)/(1+a*b),2,a,0,1,b,0,1",
    "I.18.4,7,r,(1+a*b)/(1+a),2,a,0,1,b,0,1",
    "I.26.2,8,theta1,arcsin(n*sin(theta2)),2,n,0,1,theta2,0,1",
    "I.27.6,9,foc,1/(1+a*b),2,a,0,1,b,0,1",
    "I.29.16,10,x,sqrt(1+a**2-2*a*cos(theta1-theta2)),3,a,0,1,theta1,0,1,theta2,0,1",
    "I.30.3,11,Int,sin(n*theta/2)**2/sin(theta/2)**2,2,theta,0,1,n,0,1",
    "I.30.5,12,theta,arcsin(a/n),2,a,0,1,n,1,2",
    "I.37.4,13,Int,1+a+2*sqrt(a)*cos(delta),2,a,0,1,delta,0,1",
    "I.40.1,14,n,n_0*exp(-a),2,n_0,0,1,a,0,1",
    "I.44.4,15,E_n,n*ln(a),2,n,0,1,a,0,1",
    "I.50.26,16,x,(cos(a)+alpha*cos(a)**2),2,a,0,1,alpha,0,1",
    "II.2.42,17,Pwr,(a-1)*b,2,a,0,1,b,0,1",
    "II.6.15a,18,Ef,(1/(4*pi))*c*sqrt(a**2+b**2),3,a,0,1,b,0,1,c,0,1",
    "II.11.17,19,n,n_0*(1+a*cos(theta)),3,n_0,0,1,a,0,1,theta,0,1",
    "II.11.27,20,Pol,n*alpha/(1-(n*alpha/3)),2,n,0,1,alpha,0,1",
    "II.35.18,21,n,n_0/(exp(a)+exp(-a)),2,n_0,0,1,a,0,1",
    "II.36.38,22,f,a+alpha*b,3,a,0,1,alpha,0,1,b,0,1",
    "II.38.3,23,F,a/b,2,a,0,1,b,0,1",
    "III.9.52,24,prob,a*sin((b-c)/2)**2/((b-c)/2)**2,3,a,0,1,b,0,1,c,0,1",
    "III.10.19,25,E_n,sqrt(1+a**2+b**2),2,a,0,1,b,0,1",
    "III.17.37,26,f,Beta*(1+alpha*cos(theta)),3,Beta,0,1,alpha,0,1,theta,0,1",
)


class feynman_function(function_dataloader):
    def __init__(self, name='feynman_function', function_list: list = Feynman_Equations, *args, **kwargs):
        super().__init__(name=name, function_list=function_list, *args, **kwargs)


class dimensionless_feynman_function(function_dataloader):
    def __init__(self, name='dimensionless_feynman_function', function_list: list = Dimensionless_Feynman_Equations, *args, **kwargs):
        super().__init__(name=name, function_list=function_list, *args, **kwargs)


if __name__ == '__main__':
    feynman_dataloader = dimensionless_feynman_function()
    print(feynman_dataloader.load_equation())
    for i in range(0, 27):
        print(i, feynman_dataloader.load_equation(i))
        dataloader = feynman_dataloader.load(equation_index=i)
        for batch in dataloader['train_loader']:
            inputs, targets = batch
            print(inputs, targets, inputs.shape, targets.shape)
            break