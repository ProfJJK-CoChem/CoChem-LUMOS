# **CoChem-LUMOS: Photochemistry & Open-Shell Dynamics**

## **Overview**

**CoChem-LUMOS** handles the computationally violent world of photochemistry, radical generation, and non-adiabatic molecular dynamics (NAMD).

Standard ML potentials fail catastrophically when a bond breaks photochemically and spin states change. LUMOS solves this by catching photo-cleaved radicals and dynamically routing them to **AIMNet2-NSE** (Neural Spin Equilibration), an advanced ML potential natively capable of handling open-shell, spin-polarized geometries.

## **Scientific & Technical Trade-offs**

* **The Spin-Contamination Trap (![][image1]):** Unrestricted DFT computations on radicals often suffer from severe spin contamination. LUMOS enforces a strict ![][image1] audit block. If contamination exceeds 10%, the calculation is flagged and trapped, refusing to pollute the landscape.h5 registry with unphysical thermodynamic data. We explicitly trade dataset yield for absolute physical truth.  
* **AIMNet2-NSE Micro-Silo:** Because AIMNet2-NSE relies on highly specific, bleeding-edge PyTorch dependencies that clash with older electronic structure tools, LUMOS demands its own isolated conda/venv micro-silo (oet\_aimnet2). You trade a few gigabytes of disk space to ensure the background server never triggers a dependency hell loop.  
* **Surface Hopping Approximations:** To model excited-state decay, LUMOS employs stochastic surface hopping. While not as rigorously exact as full multi-reference configuration interaction (MRCI) wavepacket dynamics, it allows us to simulate femtosecond reaction trajectories in minutes instead of months.

## **Installation & Setup**

Ensure the cochem\_system\_config.json correctly maps the isolated AIMNet2 silo.

git clone \[https://github.com/CoChem/CoChem-LUMOS.git\](https://github.com/CoChem/CoChem-LUMOS.git)  
cd CoChem-LUMOS

## **How to Run**

LUMOS requires a photochemically activated starting state.

1. **Initialize the Background Server:**  
   python cochem\_lumos\_server.py  
   *(Boots the isolated AIMNet2-NSE background port).*  
2. **Execute AIMD Photochemical Cleavage:**  
   python cochem\_lumos\_aimd.py \--laser\_nm 254  
3. **Refine Open-Shell Radicals:**  
   python cochem\_lumos\_refiner.py  
   *(Catches the cleavage frame and routes to rigorous open-shell DFT).*

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAaCAYAAADovjFxAAACJklEQVR4Xu2Xu0sDQRDGE6KgKIiPEPLcJCAWooVBbazsbNRSEPwDRLAQFMHCxkLTqJWFImInIliIjYXYpEhhowhiYW1jrwT9hszBOl6Si7nDRPcHw+19s8/ZuezG5zPUD7FYrFUpVYB9wHal/1+QSCQerTIFIplMnuj+Pw8WHeYMmOX3fXqX9RqaeDw+zLt7hLTvRfkO9go7hp3K+tDy5Jd6w4IATFIAQqFQm64j/Q9Jx3NM1wnS0+l0h9QbFk7zBalHIpEeTvkmXUdQlpEtXbpWV2CCc5j4s9TLwUEISz0YDLZDf9e1VCo1gk9mncp4juq+UqCPAtoNSt11MFCWFoOJzUhfJbjdqtRpt6HvWO9YCKqqJesd5SerXAltftPSVxMUXeq41ihzJui2Adkv6ling24Xeh2ncLa+RaPRbulzDKI5xZM4kL6fgH7mbRbo9Q9fAGPc0jjYxD7pLAnf2F5ga9LnBghuC3ZpTwvEvazjAX6McwnLUVk6v4DJrdDE8OyUPrfBqRC3AiF9XoGxhmg8bPSA9H0jk8k08wSz0lct6ONa2ZwKBI+Rl7rLUBbkONjls8AONBzniZ5Jn1OoPTIrLXV8oyH29UufW6D/B1X8M2a7CVVBPyo0YadntgXfAajdjfSp4nX5SupugMAu0rjyduoKWMwmL8rROYy6E7QT9M8Qtk1BgRxA+Vx58Bkor+4JpbC7+PwmWPyW1AwGg8FgMBjc4hNdApcOGcC/WgAAAABJRU5ErkJggg==>