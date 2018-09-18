{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Prelmininaries\n",
    "Here we perform the necessary import duties, and also assign filenames. Lastly, we import the data into a dictionary of pandas DataFrames. Note that the file we read in is large (>>1e5 rows on each sheet except the last); be patient! "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd, numpy as np, os, sys, itertools\n",
    "\n",
    "# Get dir name of parent directory\n",
    "parent=os.path.dirname(os.getcwd()); sys.path.insert(0, parent); datadir=parent+\"/Data/\"\n",
    "\n",
    "import GeneralFunctions as GF # Note that we add to the path variable\n",
    "\n",
    "# File name\n",
    "fname=datadir+\"AllStations.xlsx\"\n",
    "\n",
    "# Places\n",
    "places=[\"Pheriche\",\"Kala_Patthar\",\"Pyramid\",\"SCol\"]\n",
    "obs={}\n",
    "\n",
    "# Read in all observational data to Pandas DataFrame\n",
    "for ii in range(len(places)): obs[places[ii]]=\\\n",
    "    pd.read_excel(fname,sheet_name=places[ii],index_col=0,parse_dates=True)\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Summaries\n",
    "In this code block we simply take some of the \"easy\" summaries: min/max temp; pressure; wind. \n",
    "Note that a more rigorous QA is really needed here; so far only a manual check has been applied to the raw spreadhseet values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Min Temp at Pheriche = -22.230 | Max = 16.000\n",
      "Min Press at Pheriche = 591.500 | Max = 614.900\n",
      "Min Windspeed at Pheriche = 0.000 | Max = 21.900\n",
      "Min Temp at Kala_Patthar = -55.800 | Max = 98.680\n",
      "Min Press at Kala_Patthar = 50.000 | Max = 721.460\n",
      "Min Windspeed at Kala_Patthar = 0.000 | Max = 28.490\n",
      "Min Temp at Pyramid = -22.150 | Max = 12.930\n",
      "Min Press at Pyramid = 450.000 | Max = 673.100\n",
      "Min Windspeed at Pyramid = 0.000 | Max = 16.570\n",
      "Min Temp at SCol = -32.000 | Max = 4.300\n",
      "Min Press at SCol = 368.500 | Max = 389.000\n",
      "Min Windspeed at SCol = 0.000 | Max = 49.400\n"
     ]
    }
   ],
   "source": [
    "iteri=list(itertools.product(places,[\"Temp\",\"Press\",\"Windspeed\"]))\n",
    "for ii in range(len(iteri)): \n",
    "\n",
    "    # Place/variable\n",
    "    p=iteri[ii][0]; v=iteri[ii][1]\n",
    "    \n",
    "    # Correct on the fly (missing value flag)\n",
    "    idx=obs[p][v]<-500; \n",
    "    obs[p][v][idx]=np.nan\n",
    "    \n",
    "    # Compute min/max\n",
    "    smin=np.min(pd.to_numeric(obs[p][v])); smax=np.max(pd.to_numeric(obs[p][v]))#\n",
    "    \n",
    "    # Report results\n",
    "    print \"Min %s at %s = %.3f | Max = %.3f\" % (v,p,smin,smax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pheriche ... 6.0855268152 19.082552781\n",
      "Kala_Patthar ... 3.58461416699 7.87529626014\n",
      "Pyramid ... 4.25398980747 8.86737680642\n",
      "SCol ... 1.93398138513 8.16071999454\n"
     ]
    }
   ],
   "source": [
    "# Stratify into temps <0[?] and compute the mean Spec Hum and mean Spec Hum velocity product \n",
    "for p in places:\n",
    "    idx=obs[p][\"Temp\"]<=1000\n",
    "    q=GF.specHum(temp=obs[p][\"Temp\"][idx],rh=obs[p][\"Relhum\"][idx],press=obs[p][\"Press\"])*1e3 # g/kg\n",
    "    qV=q*obs[p][\"Windspeed\"][idx]\n",
    "    print p,\"...\",np.mean(q,), np.mean(qV)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pheriche = 189 mm snowfall/year\n",
      "Kala_Patthar = 114 mm snowfall/year\n",
      "Pyramid = 131 mm snowfall/year\n",
      "SCol = 16 mm snowfall/year\n"
     ]
    }
   ],
   "source": [
    "# Estimate the amount of snow using the expression of Salerno et al. (2015) to compute mean annual precip as f(h),\n",
    "# then weight this by the fraction of time that T<=0\n",
    "# Note, if places=[\"Pheriche\",\"Kala_Patthar\",\"Pyramid\",\"SCol\"]... \n",
    "hts=[4260,5600,5079,7986]\n",
    "for ii in range(4):\n",
    "    freeze_frac=np.sum(obs[places[ii]][\"Temp\"]<=0)/np.float(np.sum(~np.isnan(obs[places[ii]][\"Temp\"]))) \n",
    "    snow=21168*np.exp(-9*10**-4*hts[ii])*freeze_frac\n",
    "    print \"%s = %.0f mm snowfall/year\" % (places[ii],snow) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-22.15\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXwAAAEKCAYAAAARnO4WAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvIxREBQAAIABJREFUeJztnXecFsX9xz/z3B139N7b0RFUVA4UFVEsIJgYjT0aTWJsMTE95GeKSdQYTdQUEzXWmGKssWBFsWFBUFB6PeBoR7+D688zvz9259l9dmdnZ/tzz8379YK72zJld+a73/nOd75DKKVQKBQKReGTSroACoVCoYgHJfAVCoWijaAEvkKhULQRlMBXKBSKNoIS+AqFQtFGUAJfoVAo2ghK4CsUCkUbQQl8hUKhaCMUR50BIaQSQC2ANIAWSmlF1HkqFAqFwk7kAl/nFErpbreLevXqRcvLy2MojkKhUBQOixcv3k0p7e12XVwCX4ry8nIsWrQo6WIoFApFq4IQsknmujhs+BTAa4SQxYSQq2LIT6FQKBQc4tDwT6CUbiOE9AHwOiFkFaX0HXZS/whcBQBDhgyJoTgKhULRNolcw6eUbtN/VgN4FsBky/n7KaUVlNKK3r1dTVAKhUKh8EmkAp8Q0pEQ0pn9DuAMAMuizFOhUCgUfKI26fQF8CwhhOX1b0rpKxHnqVAoFAoOkQp8SukGABOizEOhUCgUcqiVtgqFQtFGUAJfoVAUJJRSPPtpFeqaWpIuSt6gBL5CoShIFm3ah+/9dyluen55rPnWNbVg6Zb9seYpixL4CoWiIDnUqGn2O2oaY833hseX4Ox7FuBAXXOs+cqgBL6iVXLR/R/gvL+9n3QxFHmM7h0ISmms+S7RtfuGlnSs+cqQV7F0FAo3nlpchSkjeuLDDXuTLkqroKE5jXSGomNp2+vqRP8Zs7zP5puPKA1f0WpoaE7jh08uxUX3f5B0UYRU7j6EC+/7AAcbk58snHzLPIz/5auu1+040IDyOXPx6vIdMZTKmap9dSifMxfn3xt89KYr+KCIWeLnMUrgK1oNGV1V213blHBJ+MxfXY175q/D7a+uwkcb9+Lt1bsSK8tvXlyBhxdsRE2D/aPz7KdV2Hso9xmu2H4AAPDEx1t859nQnEb5nLl4cpH/NHYf1Mq1eW+d7zQYBMykEzgpXzQ2Z5LJWIAS+IpWh1lje+nz7SifMxf765L/CHzt4Y9xx6urky4GAODB9zbiVy+ssB3fur8e3/vvUlzzz8U5x8MQirsPapOjd89b6zuNdEYTkikS3DCS1fAD1q2xJY2GZnl7fHWt9hx+/aL9+SeNEvgJ8eW/vY9bX1qZdDFaPX9/dwMAYP2uQwmXJHkaW9L42f8+t2nvZppaNIFaXdPAPf/GqmpsP1AfSflkaElr0jkUga//rNpfhxXbavDVhxZi5fYaz+mcdufbGPtz7xFhNuw+6PmeqFECPyEWb9qH+9/ZkHQxhCzZsh/Lth7AuupafLRhTyRpFwLN6QwyGbsaGbfteO5n2/HPDzcLFQmex0ptQzPeW2dsSHfNY4tt1/Cob0pj637j4xDGKCGtP8cURzLd9PxyTz71zEtny9567D3UhHfW7PI1r7Jlr78PYD6adNre1L1Cmi/dsyDn78rbZoeetpc0SZ76P4y68WWce8zApIsB9s3JcCTvuupa/N8zy/CLL4wDoAnDc/66AHsONmF47454yzTfUNfEN1+s3lGL0X07ZQXpZQ9+hEWb9nl6h2t31mJnTSNOHNWLe75FrwTvXT/yfiUA4KYvjpfKyzxIuPTBjwAARSn5NnTP/HWYeXg/6euttGScBf5Nzy9HRXl3nHXkAN/p+0Fp+IrIaU5nsHpHref7LnvwI4y68SXbcbM8W7Xdnm51TQOqa/kmi6A0NKdxxcMLbSaRZz7Zars2nz5Qt760Cgsr9+KD9cZI7dPN+7F5bx33GVpZsmU/Ztz9Ts6odNGmfZ7Lcfpd7+DSBz9CdW0DWtJ2gbi/XlusFM6krZ0S3tCBw4G6Ztzx6mpcfP+HvvNn5ikeTy2uwieb4l+NqwR+gbLvUBMO1Oeu9Nu0J3o7N6UU98xflyNwb5m7EjPufgdbJDvx9gP1aGxJ4921u9GcpshkqOO99abJtAff24jNe+ow+dY3MPmWN4JVxEJ1TQMamtN45P1KvLV6Fybf+gb+8Npq4WReEJPO6h21+MNrqz0tGvK7wCgtcd+OA9r7dBLyb6/ZlTXHyDD5ljewttpu4x4/oIt0Gm4QzjyA7NQAe3eNLf7NMs2cD1q2HEjGXVQJ/ALl6N+8jgm/ei3797trd2HaHW/h+aXbIsuTUoolW/bjjldX47uPL8keX7RJWyS1X2KpeTpDMeW3b+bc/6c312Lq7fOxcbf2weJ12gP1TfjNiyvwlQcNjeyjDXtw6h/e8uRhYSaToXhy0Ra0pDOYfOsbGPvzV3Dby6uy5//85jr8nTMPE4Zmf/HfP8Sf31yHWh82Z5n8zVfsqs0NPWAWxI8s2IjyOXOzz5wnAN9aXY3LH1qIv761TkubaPb9Qy5lv/P1NdnfN++pQ0s6gxG9O7mWXZYg875+vp1V++pw52uGlxbPJTZLQoM/JfBjpq6pBZf8XX6YSCnFzLvfwWMfSm1K78gG3Yvl443RrVD99YsrcM5ftQUzbp3dCWZ/fn3Fzuyx93UzxE7djMLrjEyZOmjqZL96YQXW7zqE5dsO4CsPfIhVO7x5aDy1uAo/euozPPDeRsdreMvng2huW/bWob4pjWZdsKYFZgF7voJzPrX/P7+pCXH2PimlePT9Srz8+fbsNcwNsXKPMQqbfKv7gi/2PqtrGnDSHfNxS8hea15k6k3PL8f/Pfu5PQ0PiZz4u/n4k/68GL99aaWjc0IS6wOUwA8BSineX7db2Kkqbp6HC+77AE8vrsoKMAD45j8WuQZZWrWjFrtr/QWAKp8zF8u2HkBxkdZyRRNJgNb5yufMxXNL7DZpEYs37cPDCyqzfy+tOoA9B3PL3NiiLcy59+31runxnqRXQco666eb92PBuj345XOGh8eiyr0onzMX81bsRPmcuXiBM/LZp/v2W+vhhz0HG7GzpgH1TensSIXH1Nvn42uPLMyWvaYhogBcATTMXz6/HNf+6xNhUrUi7VanJU1R35TGXv05v79OzhPspc+3Y+rtb6K+KZ3td3sONkqN5JwE+CPvV+LfH23O/s1raX9+Yy3K58zNMdXsO9Qk9Py5750N3HmApGZ3lMAPgeeXbsMlD3yE/wpWKe4+2IiFG/fi58/lupW9vmInbntlJXYfbER9Uxort9fkrHZkdsjFFtupk017/qpq3PD4pznH3l6zK+vX7Oavzs6bG78b1bUN+DInkNnEm+fl/M2E1wPvenNHDSsmivn+1/QRBPPjf/qTquy5+qY0dtU2Sml3vDLxTCoTb56HY299A9f+azFO+f1bSGco/vLmWuzj+Mx/uGEvmnShcvLv38KvX1ghNzrRyxKCC3sWa1q8+vp9LSu21+CwX7zieTR447OfY8veevz93Q245IGPMH91NSbePA/fePTjnOu2HxBP3Nc1teCIm17F/FXVtnNMeTNX/2+6omI2a51659v4ncnMx4M9w/fX7cbJd8zH6h21IITEHtQNUAI/FLbt1xqWSHMT0ZymqLh5Hr78t/dx5h/fxY+f/sx2jdlP+unFVZh6+3x8yPGN37j7EJ5bkqutUkqzE2oLXUw6JdmRgHxjrKl31kJ/8tRnHCHhLJFEfUB0jglO3kQd+9iZ3RWZ3zw7VGxy1zv/vvcx6RbjYyV6r15HIgv09/ju2l34/WtruGYEAGjQfbgpBR5asBEz737XMU0rvKfrVbT837Of4+55a7LPJ4yPyCeb+RO+NfXeBH5pcREA4ONKrS2zOYgF6/bgsgc/wsy73wEAfPs/n9ruvev1tVi7sxafVx1A5e461Da04Hev2AW2qPmbH0WK8N1gzdQ0tOD9dbtR35xG5Z46NDSnQYj/D2UQlB9+CDBh0ZKh+KxqP3p1KsWAbu2l72cNaIVpFWA6Q/HGyp04fVzf7LHdBxvR2JLBYr3jrNlZi0Hd26NzaQneWlONju2K0YkTFTFDjRWWrnUp0nQAnsucE6KPw38XbfHkeSESluwMb+KQfSR5q0yZLM9QirdWV2N/XbPNZ70oRbBs6wF0bV+CZVtztel5K+0aoF+KUyk0p42l+gcbW/D6ip3o37UMhw/sKrx31Y4ajO0XjheLm/xmI7yeHdvp1+txaTjvJ5uWiwR7b+1u8QUAfvDEUgDA4B7O/YeZJ3lt+l2XPOat3Il5K7XR3R8vOsrxuqyGTwjOOXogFm3aiz0H7W2LECL8ODAueeAjPPy1SQC0NqdMOq0YtpgjnaH44l8W4Pjb3vR0P097eui9jbjqscWYa5ocu+DeD3DCbW9mfdp3HGjAib+bjxc/34b73t6Axz/ezO3JGUoxqLvWgUb3FXtBFBHvGn6lx5GNSFtkyhJvuOt1CMzyycZFB3DFwx/ju/9dkhX07GdxKoWz/vwept4+31Mebiadxz6ozAlVwJQD9njTGYpv/mMRzvrzezlpjOjd0ZZu5e7gvuleYdWTiUvDPgZO71fm9THTGpus3rTnEMrnzM0xu6RM7zMIuzkCnGF2VSXQys4rf4qAu8qam2aatTntbzVp28p4c5U24ccWi7hNiHph015NiJptvBt0wcrs+Z3LSgAAew42oV1xCo0tGa7mkKFGIyvvaRckZtw6LY8bTC6UwrQ9NHDzpUaYW3+QrIbvnI9fk4XoI7S7thE/f245rnjIsC0XFeWal5zMAby5lvbtinL+XlS5N2cuRzg6CihceKYy6zlRHpkMxV3z1jhfYKFZf1nMDPQ/kxMBU7DcBO3xI3oKz7N3Zy53Q3Ma66prs31rwqCuANEFvv58zbkWEYIMpbidYxaywkJeUEo1Gz4oPq7ci5tjDLKmBL4P1u86iFeWbcfXH1kEAPjTG1p0QPPKurvnrcEX/vweMhmKRxY4u/UB/Em+5hZd8yxyfkXMVNGSzqBdcQpNLRlux6zaV5eNjii7tNyLL7ns4pSsPVg65XDg7XzEBC0rSxjBuqyw5rDPFMmTZH/mavoylBbntoXz7v2AOyKRqYpIgIvw++HY6HHR367aRpTPmZuNR2MurfEBFxfG7A0ny/efWILT7nwHBNrzHt2vs60vUEqxbOsBNKczWZOOjMmUKWzpDM2OGs6/9wOh22/YKBu+D079w9vc400muzcLETtv5U7cxAlTa4bX99iQskjQMTNUm2RtzlCUFqdwsLGFK0zNy/7DlmtebP0y2rTfSVsRKY6AyAp8vTC8Msl89ERFIoJrDBOJcdav18ZzS7aitqElO+Gej/gtGc/lkfWJoKaRm+fa/f7ZTmoNzWmkCAGl2rsyv5v1uw7hS/cswDdOHIZUSjsnUsysZFia/oodCKXhh0gzZ5FMfQDfYO2k86kMpShOpTQNv4hp+OK8rBH8ahuac9zivHaeD3xE0RQJUv6koEh0yudntrh50axFiMwKPKGe/cDof5vDEbg9+wylqG1oxg+fXJrjn//C0u2ubrRBq2u8AXtKm3XtPftB57xfvyMKhjmkA8/rKgj7640RmKEcaO8vw7RxGO+HrYlZumU/UoQgTamnj63WHpL5OCuBHyKR+NVykjx1bB8AwJAeHVCcImjJUMOG79KOWjIUm/YcwpWPLsKSLftx9j0LuG6gvHQamtP49QsrUGsSNm952NVJ5vmEqeEzwSMzB+DXpCPjvmcutygXt+qlMxQPL6jEU4ur8EBOSAe7LdqxTB6raXirOF/zxKIq55MsX2/ZmvLXfpojeFpNOkF73c4a88I6w/yX0s01hDivPyjSrynxoOGz0buatC1AZDSb/yy0L9gS3XXR5CEAoLl+6hemUtpCDjczxNtrdmFnTSPmrdyJnTUNWoPNUGw/UI/314td2h5fuBkPLdiIv5iWjz8oYX+0Nmy/C5r89o8UR9O2egP5FkiCUpm9g4xjyDnmpU6XPbgw+7G1fmjMz1Q4gvL5EEXbBcqsgI5gikRYplqfK5StGj6F1qfMdbS+zwylWVdRGVjaSRh1lMDPU2Rsw1bNjkKuY23YpQXHymSYFkPxt7fW45uPLhIOk5mr5qGmFpxx19vchV9BeH7pNsdY7EDYwirXhu8XUZm8zlXIjIDKSjRPHa9hJqxpe6213EfaXqYNuw7i1pdWSr07Xp3E6zKcz930vD/Pl6wHkK7hU6ptxpKhpo90VkNn11C00zX8604e4ZoH88NPQsNXk7Yhwp2c85mWSPDmaBgB8slQbWSQzgB9OpfiUFPatAEFJ1894427D2HNzoO4yGOscMPGa2fJlv34zn8+xYzxfW3nDK3Yt8R3v9/ni5KxI3PXFHDMMDK1Y4Il5z4Pj8Xr9806B8XLyqxwWPO48tFF2LD7EKY6bHhivp7/EdSvEeZrv/FAvb89jlk4hgylWe0dII4jzhQhyGSM9RXMnVME1dMuSJMOIWQmIWQ1IWQdIWRO1PnlG74VSBmNiFo0VOvfLmVKU4qilK7NsNXCguiMrGMVS24iwSuvlr9RxqVb9uORBRtR16RNHFftc95O7vp/25fKi8guvLLkb/7dMOnwJhrd8+AKfEu+ueYXi4uf+XeJd87ek/VSWa8Pr0KmQWKbPla/TzfbN/QQ7WDFELXZl5ftsB2zhnvg1WlDwD2OKUWOlw5A8bPZh9muIYT1I60wRw3ullM2HumMN7fnMIlU4BNCigDcA+BMAOMAXEwIGRdlnvmG3xfL65dHDtKW3ls1RraQQ8tPvkyZDEURIUhnaFZDyXqNcFosy9ar+19tI7On2mt19j0LcNMLK7IfEdEHxy/s2Yi0cb8f5heWbrcftGST66XDjvFuk6+7aGQg54cvnZUlY+mDOfmIF4U5n1uyZX9OOrn3Od6W9Xn3iybwc80vhncQsmUq0ufOvDxQY7K58Gz4kwGso5RuoJQ2AXgcwNkR5xkbHSwrH3nvL8yXOuuI/rZjFgVfqt2VFJvslCmStVcCcrsfedkXFDA2gRZ5tBRl4xGFv/Ezz6kzrOG0dVcxHjwTnMg0IsIqQMf266ynK/dO/Ap6kVumcB5D4hq5V2GeubLPXVk5bngPqVSdYKYcs8+81TuIAIYnj4e0zSYd1u7jipwZtcAfCMDsglKlH8tCCLmKELKIELJo1y55F7984EtHa1U5dphz4wrq7y0TTExrmhoyHZ9p02mThi+zXJ2dYdEKvSJq00XWEYaJoB4e1PaL/ZzHb5gYk6eHY5kMm1L22M4a9314s++XAt06lGCyoO3x8/V0udT9oiR5nkr2NN0LtVu0JwHn/qBB5jQlCADz0jF5wJkX7WVHAZb2I6oS+0BQAN+ePjJQOb0StcDnzrXk/EHp/ZTSCkppRe/evQNn+MSiLdz41lEgIyPC/HLztENzGSin4fFgS/SZhm826bDNHUSTZIz2Jd4Ev9FR7OfY6kne4jW/GNolG0IbWLXESG2qpoxtgsH0+4X3iSfB77nkmBy3TsoEhw+/bq/19SuwjXkMkUlH+8nb49ZvmYKSocaELNPwmVLAPiYnjOwFoitMXp5nOqPH0uHMKUVN1F46VQAGm/4eBCC6TVUB/O2t9ThiYFd0aV+MljTFscPFAZTCIAofY2sn6tul1FGLMJsQZYpy1WNaXJ0M1QRtSyZjsnPr6YhcCuF+jeg+HsyUFIVJh+XrdWVmcG2Y2vIVCYYdLhq+OawupTR37saUrFd7twzWidKcc5a/N5m2OsyOdkQjR59l433Iw4Jp+GmTDZ/Vnc1htSsiKNJNol76Aq89/G/JVozt1wXjQtzEnUfUGv7HAEYRQoYRQtoBuAjA8xHnCUCLZcPb2CBuwuhoY/t1xoRB3bj+5GYPB1kbPoOZcnLNKCKTTrDK8FZtMlMOC+/AM+lEiejj5bu+1klb3iV2i44ry7YdwEA9zPWk8h6+Fm/5hT2Ljyvtm5iIhDnzunpqsfNq3NU7a72Vxfp8Odmv3O5t/2IrLWnzx5S/qNEYYRnHvCwqNLev7z+xFLP+JL/JjV8i1fAppS2EkOsBvAqgCMBDlNLlLre1GrLCggliTtcLK94HwDEHmGb7jVPyEp8ykw711lDDHNB0KSvGvrrm7B6ovDj8gc0tPLuz5Riv/n6/PcwsVszxmRd5rbj5Zt/3thFOoU+XMsDjyM6aVxywKJI890rGCo/CWea1iHYpG9O3s+tHZrMedjpnYphjjmPvzNOkLfQRgWS/C5PI/fAppS9RSkdTSkdQSm+JOj8gHo3HjNjnNpzSOKVi1jA8eofpk7a5GpqX75NoVSwPw7vBKGSRaQIZMDaJCBM5Td3+4JZtPeArPybw2dDfnL81l5wl+55s8Oyn3XuFR/DRmeBcoJSBuZ9xXFs9wF/YFgzmgWcWztmNV8z9TQ+7IOp3s47oZymv/LqJsCm40AoxfzBdCfpSNS2CXyuzCceLHz4jTZE16Vjjs/MnbcMXGuZFYFr+IXYDSwfNKYvE7S/6FERslMI+Zk4mOKeyyZI16djScG4FfgW/eNLWV5KhITKZBUmTvaoUITmecMY1mmrPBL8T1m1Hsx8KGv9WhwUn8M1E3Q6lGlVIhdC0Ar4Xt2HLpY4fB36abOcdc1rMzm5PJ2gnYlq8OelssKoME/j2+6KZFM/9JVS3TGtenGNB56Y1U4JJs5R8SKzeq3Z4tZuLJl0TlvgekXlUlz+0UJt/0M01vIlZs3ulSHLbbP8wK2ry5Q6Dghb4cSH2aAmuURHLz9z+RbjX9OrUzjUPJoBl3TmDwpMLbJjMtOIoVh9yNUDL0Ug9rTgFCDqS0UwJJoHjWpZA2YnTji5pfn4WN1TRyDFwXjDZ6U0usSzjlIRtxnnuLX4KQuCv2VmL2X96Fws3arvVJKVxcE0HPotinyC1C+VV22uw+2BjdmLMqw2fO0yN0FbLzDbmPK2bWUTglZmYycE88mJYQyv4LZtVCFm9RdzKFCZJK/hRmHTMsP1nzW3VvPrW7C4r2/0MV894VfyCEPj1TWks31aDg43Ntif+9ppduOj+D1AtsYoxCsKx4fPPsa0Tl2+rydZbFKvFlrbZtqz/tO716nS9HwxfbnvqhkknvJ5qGMBEk5nRw/PSCWvzDt6zFI44Ayog3HOJ6Kpu+YZTJmZnty6Sytn+EGLBzVtsZzWlxkVBCHwRu2ob8eGGvdIbbQeB1ynCEmAi1y/zENeLCyPzLvAzMvADr4OywJtpgQCMJGZ9jL2Na8MPmL8maAyTnGsZAua37YBzFNO4NfygJixP81yANjGbcx817PoSk6+2/JjtP4GhUUEJfJkFGVEQxQ5DOelbPHG4QjGr4sPxGm7aHj8QQTA2nTbSyQZti0DD90KY2VJLPXkdOyM4J4MmaExzNxFbBm5/ZbXjucRMZpz2xNhzyF88fB5El/jGzmnIMemIRuHa/bkw754kKAiB7zRkigvDs4V3LmDapnk/pyrxTDNetD62jZs1LafrfT9ZTuLWiegohIdIwY90MtP07hjGIr1gZOcKTS9DSvNNyPwSKkFHR16yosZoyuy6zD645mu8pAm4fyiioCAEvhlns0eUeToLy9U7gi3xNqfvWLesacabWybA+1hG9/HKcARgNu04JC/vlKC+vrNjphZO9lkbfkCbTnalJqdtcOdfIhT0ydnw48nDmGBlx6hu5qFZH30RXIWU/RIzBSfwrUT7AXUXFv/8cHM4OQn9oI3fDSujl7SNGw0tXmCm8pB2bj7Od4rkn993aHOjE5QlGpOO8zWGScdfHjIrPEVlC5PkTDpaxsu3BVeqRGTNNtl8NfNOVvun4miZKctLYt455lFDXBSUwE/ePSz8AuR0asHKUeaP7c1Lx3AvM44Jb8jm5YcMRxDK7EYVlKDatF9Em4UEra5Xd8woidssYeXZT7d6ut6b+cU0MWvqf+bJWu9OD8ltYl4QAt/2lRSYDsIk7hfm5GZoFvSGeUnOhm81CWQFbxTzEYJzN89dGTB1f/lGqfHyvjPs3QVfeMX5cEqOAsMmqQinfvHqkZayTMxm/fBNmrowP2uapg+EsuGHBHdmPGQiNTtz/naetDXMAyIzhmNekrb/oELKHN3TC0E7BV/w5rLaY6gBEdn6cZ6X8THw9yxy88l1z5S6J4JGm9AAKj4TOMn1iAKMEfWhxhYs21rj4oevnRveu2POcaXhB0TrALkPPunhZhhkLToOk8MUdk1dpjfQnBs0Kvc4h5VN2nTgFavHknXxDGDYfxdtssd5D4rMyMK3Dd9iOza/RlGSSQnnMBGFrAg/M7Prq9H/2MKrj/TV/d/5z6euSZ08ug9L0piDiaLMAgpC4Ccl1CP1fOBMKjrWM9v+jdYpU7Ic9zD92D3z1wOwj5DCgDdRGcery1qpPLowBs2PN3cwYXA3AMBoffNx/xDHUR/vI8IO7RHtDdva8Nl4/CxONGeX0dV93ohXtPLZWLdlmITiJuotDpMlxCd6oL4ZXcqKc16otT9H9gJdvmh+bfja9ZCWftTy0ytJjRDiXszFcmOhLwDgyEFd0bNjO/Ts2A7dO5RkQ+b6LZl10nb1jlo8/vEW1/se/WCTzxzzkJheKzfapY/7rR50IjNtVBSEhm/Fj7uaiC176zDhV6/h4QWV/PyimPgz/Z7OZKQamS9vAf0GmfuCLrzieenIEPRV8rKL1u+flx//p1+sk7a1jS0m98QCsNsICKp4eF0kxYt2ybY99JMfm+xNgoIS+KKYFkE62BZ9u7PXV+zkpx1hB3t9xU4s21qDVdtrHCdkzeGNffnhc47xTQXU8ZxcPv6ek/+OHb3Lpyy5H1d5YeEEMdkJbHM63C9coOzaLNbJWgD43SurkKHeW7PhMm1ysojZil/YJh2EM2RK6UE0rO5nUQoSVuy3Vu8CAFTXNjo2jtwhorOvvu0+yjRFuRGRKJKmDNwySSTGnoFf4pb3TqKAcM75j6WT+9PMpj11vtJsLRijpHherNUG/+EGbaJWtv+nsvfrfRPsvcX/FS4oDT8qivU31uIQrD2KdsfdXs/RLdM4mbP82wWe7d/Iyp5Z0HrnwrvXAAAgAElEQVQmt0+BPd+4S2I25TCXPgAYP6Cr4z3tity7J+9D/cGGPd4L2IbworCIdpGTbc68bS15UWoHdC3zUDJ/FJTAz2q6IXttFTlo+FZ367iEiM2DB+Yhoo/0Qr7O8f6YNTNGvrgimjVF9ggumjzY8XqZjzZz71NEg6ipbt5rH0nxvg3WQ0u27LetcAeAsyYM8Fw+rxSEwDc/ZKstmxGkSziFJY67m4mEOas3i/vf0CwR/9805+HJpOPTTubTohOYuG3aXLdIB6UgqA1X9lUU0kfBGpzO+/3ecHrG767dLZlAzg88v3Qblm2NNv6PEwUh8EWEOSni7AYfly3RbWhJUdeUlk7POiLKzcs5H/9eOloC2w7Es/tYjt80csudjKumYXlnbebiv3/oK72cNQWFI8tjocnDZkgU9uBnIpwcOwDYOo51hXscyk/BC/wwcDJB8FZwRonIA8lvY3GKBSJq4741q4QEk2gDkkjyczies9gm8HyInmawZFolQV/dtv3Ou3fx8vJk8+ctxhI4Wzz03kYPqQenoAR+WD7OjlgnXyLKRjL7LH4/PF4bc1ixdOJm/S4tXIQ590jd8LkxdIxjPPutX6RNOgU4CvBbJ8/rQDx0Et4oPOulY13ARYGt5o9PDF/vghD45gdp/j03dkr0Gl2SncqPWT1nstdD8DTfbpmcY3EsQJn7ud3jKYmJXK+ueDLtKanNsPMBv6ZUb4qLEa9oXfVB16tl5tmMlOOnIAQ+D9EK0rvnrcGdrznv0enE0i37c/4Oa3gui1Os+0DFcGihQrdMnzI6bru50CyVoEnHukpW4Y2WdLCH5+Vjb3ad/O3Lq1yvF0bNdLs3BhW/wBZeyb3JjzbsDSWGd/xeOqLG5L2xMKFTta8eu2rdg2qF5ZaZDyThh7/nYBM6lZZ4DmnMI0euSCSVR48+MPXNmmOC3/YU6YJJnlumiyk2TgpCw7cPleJ5kPb47tHm67Soyq9/+38WbsZy3T3sNy+ucLk6eAPNF394IP7O1pzJYNGmfXj6kyrpe04a3dv1mkII/+2VoO+ud+dS+bzgzewouvI1kQdPTEQm8AkhNxFCthJCluj/ZkWVV26+/ON2H3oPtlS38wkLMr/B4vbVNWP1TvmNP9hC42c+8balHIPrDeQrJTlEo54oPz689mA2Q2jb47mnc8yQbq7XxB2LJR8IOu8zrn8X6WtF8bl4iIq22RLywtoG4vh4R23SuYtS+vuI83DEMaZJSA82ZhO+I1Gs9OX64Qf2JQx2e1C27jM8IpJwy/R6jYxgkzUPJeUhFSV+q+S1/3u53ovPfhKL4QrCpMMwN4AwP5aODcunKcUvToHRKBJcseqB9bvcvRzCJG0pcItJrZdeJekHbuyeXLfMsJBNaqsH3/NCYJrAHOZNKHvrW9wNUCRTKISFV9cTQj4jhDxECOnOu4AQchUhZBEhZNGuXf6iIvLiTbshc82sP76LRxZsxBMuG0vEZZue5xSemQJPLq5CTX1zIt4nssxbWR1KOWRZHMG2hX44cWQv2zGZ9yQjl2R3TpKZlG8reI+HH44Nn5d23AQS+ISQeYSQZZx/ZwP4G4ARAI4CsB3AH3hpUErvp5RWUEorevd2n6QKA9n3t2J7DW56YQX+u4gv8K0xPaJ+gc8v3cbNZ/fBRtQ1pUP98PAaeRRmkEKcdLQ+pYsnD7GNPqVMOiF6ZeXThHlY1DQ0O54Ls7opD4/a68ckbgLZ8Cmlp8lcRwj5O4AXg+Qlg21ilvJ/Dy0/i6TPh402mgP6KIvIg+q1Ctwm42Q1RtFlj35QqV3TBidtD9Rrgr62oSXyvCjgSYoL34fl1JZ9uZO4cSg/UXrp9Df9eQ6AZZHlxXnIRiha5zgWQWEdO27t6QdPLnU8V9cUTid4Z43dvFaIE39R8PnWA+4XyayiNf9OgO9MH5n9+0E9BkuYYRoKCVFbZadEdn5zOkG9dJwEedW++OdVorTh304I+ZwQ8hmAUwB8L8K8cnDrANYPxM6aBlTc/Dqe/dTdR/rzKqMzGyadXNNOkkQZqiAf6tca4b0RqVj3kq8yH0aW+YbbIylKEVw9bbhUWp5i6YhOupQpjjmnyNwyKaWXRZW2CD8CL0Mpdh9skoohf/+7G2zH8qm/RTkqjGIks2ZnvJ47SeHWRk47rI9tUtsWbItzH+HsaasQ4zVooBezGU/8sLk3N9jWiVFSsG6ZnLOWa3MXwgTJjwnCfOh4UdoBlSbpD82Thlr+zr2me4d23PsAYOb4fsIWqt6KN7ILFSW9+WT61GH6Yq4pI+weWfm0x3BBCHznWBX8Lzm7/rXlO/Ctf3+C5nQme70XjDnb/Jm0jZQCr150CALRydztEDSPpazmVuzI7uksg4wOdeywHtj421mYOJTrfZ43FITAt6K1/9zXVNPQgp8+8xk+NG3wvH7XIcz9bLtvQZ2P/SxKDb+QtsmLk+rahpwn52VizzjPv0DWDz9phvfqGGt+omeSofKjetmwJYQY7+jqk4bjL5ccLZV+3BRUtEwK5xn1huY0/rNwC8YP6Kpfy78/+7tDi6nneMHkU4eL0k0vI78znMLEL55bnvO3tvE4bMecIARIZyiWb7N7/2iWiTxqgHmCq5CW/FBSyY+D+ZqfzjoM2w/YPXDy4T0VhIbvaXWb+T7BUNlpgjJ3Yq2NmHJ08qHBFgp+zDDzV9tdZVvLpG3cRZQS5rKlkhAw1sVZ+fpOCkLgixB96a2nzC9p1Q73XeWtQcvy4SVHatLJg/oVArx3lBL0RDftvxBX0QZF1Fa9bEIj69GTskh8nhKYD++poAS+8YyD+8W/ttw9dvUbq6r1fHP98QuVfGiwhYL9Ufr7Utc3pVvFpG3cZXTri9J7AYPiy8cMQvuSInF61vvy9JUUhMBvbNGMyws37jWtsHW4mAlnjruV33dkbEDiM4FWQmsQLK0BWe0y5wYH7nx9TdDitEG8KYQz7n4nu8uWIxIfELf+E4eHT0EI/CbdrfIzwZL2nI3OQ7Z7sNeYD7Z8tdK2deL3tTW2ZLCjpiHcwkRAPtnw2USsjAJDKbBWYvPyQd3au+bvltu5xwx0zScoBSHw2cOV6TM5k7bWOyhFQ3MaDW5fc1v++SMKo1xpm0/1bM0Qzqof0Xtze6f765yjRirsUHgx6bjz0BUV+MqxQwEAb62uxg2Pf4pDPrz54giEVxACn8H1b3Z4iE4v/asPLsTXHv7YU775ZNuOdqVtdGm3JXgmHaFzQQHEkGb1/cVZ42znenWS32NWOr+QrpFh+ti+2UnbddUH8dySbdnFnF5o1dEyk2Jd9UGs3K7t08rTSEWz7llvG49NgV1f6ALxbU4ETYUiOBF0HKFJR1+vE+dXIU8oEIFvvJXGlgy27q+3CXXpr6ePlYtGXHxv90VBW4yP3urghDQWb3bS+rEqUSePiWezIycIIZLhF7x1ajaP56cfxvGeC0TgawgDTFnfm0D79yq39xxsApAfk7aK1kFbbypJfsTCfvTmOT82yvdjnlEmHUlEnce+3y01uW46bJzisUUwl6186MMFYO4teEQB/bjXF/g7jWQ3OkFv9Lrwyo2PK42wxkGUPjVpKwl7xGYB/tqKnVi1ozb7t0yn0Wx72lDPz6SL8mKJlgFdy5IuQmhYBVI+yvThvaMLeBb1RPSnm/c7nivv2UE+lo5EXmzLRSD/R26FIfB9uGVS0/XW+DqUAo9/zN+4XCZtEVefJLfLjl/yUXCExaRhPZIuQijw4t+wVdvc6yMujxP9uoT3gaUBTB1+aBF4ULCPTViyuanFUA6Z0uernsqk4w3Zh8wu4620ZXuEFnvZqp7dnwdf90Ie/hdK1Xj18Lu/qew2fX6Ioi3xlLMkuo1s1WRG7eZvSxBPPTVpK4nopcjEuCCmc2wFXkW592XOUiadQpFaCZAH39PQ8BRZQSB5Tx3bN3hhnPINsbF6WXcQB9ooy/0tTB3l7k2UMUn5QF46MTyUwhD4+k9Z17aqffVYtvWA4+SZpun70PA93xEOXzl2SPb3b//n04RKoZDFuuVhEHwMRF05anA3AFEHA0xO4hvBDt353umjXa9Jm95lPozyRRSEwPfK2uqD3KFX1r7v86Ul9bIvnDQ4+3tzOs9bXAAKaXD01OIq6WuFYRcieCjMnLlg3R6XK71jOFiYjlk6zrAId8e6cdZhnkIryJDJEfhUVxq990Nl0omA3Ala++SNtiAjuocftetV3y65y9RnH9k/0vzipFA+ZWt2HsS7a3eHklYUZoCkg7FVRBg1koVA8ByxlMOQHh0AaCadDbsOYsveOmQokPK5KY3yw5eEGjYdG178m40l1/5aQj7Ewz9iYLecvw/Xt3Q8f+KgJIqjCANBG/bSVE+RXN1aFIGdyKrFEwC3f/lIPHvd8aHnJSJX0Afrr2wklKHA9D+8jam3z0eGOm+zmg8UhsAHmyiRuNalhzAbvq9yJC/vAVCM69/F9FcANzFF3uNlLkDkqmimKIbGQghwwaTBOHpI91jUJLOZSNuYXE4LFz3fkiJNfHYsNbYGp9A0fD8LsJSG7xHeA9u6XxueVu45ZL+elwb8C+7l29y3RYzjpXqJGqrIf0TvLkOBru1LpNIxm5G+L5iMtG7XZ+bKE4dJ5WUlaV3oksmaY0N2lX0IaXZpX4xuHUpwnmn0nGE2fD8mHbXSVhLBw92h7x4v4+dc15TG/NW78LlgI5WgRP1K82OUoYiLDKUoKzG6sfl3ESKrjWgNipfm1bGd87aAORsSeUjTLxmOJ03QrrJm50Hsr2u2LLxythK4KXtKw5fEq1umcTD36F3zWvd2cXdeMAGAg4avFPxWi+jdNTSnsbOmMfs3pXJeLqXFAmEcUmPpY1qpa2wDajcxxqGjVJRrq7RZqCxZLVx0DQupYA7DkslQfdJWu7GkKPdZtitOVuQWhMBn8NqpF403FZPtMkqUgh8dvTuHv1GHCJm2W9dk350taDnjiAkVlwLSq1MpKm+bzfkIyhVAxmPJPMnNbPjsCd7/1YrsuZ4dSxM3rAYS+ISQ8wkhywkhGUJIheXcTwkh6wghqwkhM4IVMwSYC6apLVsfvvlvtgvPfZdNjLZcIWAd2Sh7fTT88Az3RThRIHqbopXj+UzcbdQYWRCTScf9w7ZpT53rNUUpgv5dy/CFCQNsNvwUIRjbrzMA9z1rW8NK22UAzgXwjvkgIWQcgIsAjAcwE8BfCSHOY8iACDcsZmXinLM+X7OGP+fMsbjlnMNx6tg+gcsXNeaG66SdKZNOcPLxQxrFHgwiwRP0CYiK+9Mzx2rXBMxDhFY1+YlVmX5TRAi+OqUcZ4zrawq9bHgOpgjBaYf1BXFJL47WVex+iTOU0pUAt4GcDeBxSmkjgI2EkHUAJgP4IEh+juWQcD1k54RfddP9ZSVF+MqxQ0Mf3kYpNIw4//ycFQFJ6BGK2rWfyUHtPvn4U37JDZBmDaZjOhdDJE1z7l42OpIpUipFcO3JIwAAiyr3IpUyTDrmCdx8MLdGZcMfCMAcX7hKPxYJI3p3AgCcfZQ9CyM6n/3VWY/xnBNa2wbS+dCookJ5INmxKiTaKnH3Nutl0yBr+lERxwiKuV3LzteJ+v/M8f1sx6wrbVkwRhY/SehYkg9eOoSQeYSQZZx/Z4tu4xzjthVCyFWEkEWEkEW7dvnbJHtAt/bYcOssXFAx2HaOaRZMmJsb+uodmt/8tv31eqGJtjlChMTih2/6PUrt6c8XH+3rvh/NGBNySeIhqU8/T0iwVbNRfARH9ekkKIs/bNEyPd5/x3lH+szZnn/GJIDdEPWbP1wwARtunZVzLEMpUgQY0bsj7rnkGIzR7fcsGbFJJw9s+JTS0yilh3P+PSe4rQqAWfoOArDNIf37KaUVlNKK3r39b2zstFhEJPAq9QmZfXXanrQpArz2vWlY9ZuZvssBIDu8M/O1E8oDpSmLUxuOoimxZ3r4wC7iCx3u85tfHHQps1s782m0d/30kehcWoyJnLgzQYvZiVP3sDDMrybPFolNQzqXyS0uc8VjrCzRdYTY5c4Z4/vh+lNGomenUsw+sn+OxxSvb04bbci8vNDwffI8gIsIIaWEkGEARgFYGFFeUvA6K1swUZxKZa9pV5xCWUmw+WXeRBpbhh0FNs0gJsEUv6eF9vNnsw9zvMbcgcLiiIFaPKLENHxOxhOH9sDnv5qB7h3bSV0fFr7Djlj+jutZ8ubuNA1f1qTjLb9po3vjihOG5Rxji7EAe73HD/CmLAUlqFvmOYSQKgBTAMwlhLwKAJTS5QCeALACwCsAvkUptTsMJ4C54ZX30sw3PTtpnSasjiKyD0bR0HO8dJzyzR/lNNQPxa/PHp/z99RRvUJLm9FJj5WST8+QYStSCCaeKD/kLAyENaprXOhOOh7CH4j6stxzomB7ZXPS4ExeR0lQL51nATzrcO4WALcESd8P875/EopTKZz8+7dcr/3lF8Zj9hEDUDG0O849eiC+ctwQ13tkEDaDCKUGa4DmHOKZM0hOEvJMGmGTdBTUSLYa9Jmfp6JwhNkpY/rgrgsnYNYRRthuuafrftWr3z0JM+5+h3+S5v6qLY4KZsP3guFB57xmJo52Fp2xLiFG9umc8zezDzKt2/wVLSspwom6RnjnhUeFVoYkNUGb10YeeraE9Xx6dmwXSShfJ/JSw+d5lulCJEX87bEaaTUJcM7R0YTqZhOkwuyJtwBnYTwLI6wEvzzW66KkoEIr8DD7w/pl0c9O83R90gt02lK0zEHdo/WqAsSuvUljLRPVFxUB4klmmcWKXs+JSUbzYE/A6ocfhlumrEzJxu6BPVY+sVwXNQUv8BnGi/f+WFmYBem8OA3BvPIubGQEURTaqW9vmxDL0Km0GJW3zc56CkWpJcWt4cvkJ7omisGP3yQH67tDlaQ4IidgEDPhfZa/CUh2kxKZLRzDeoTZPmrzr7B7K0VJwQv8OFbxWYlbD7R+xJLUQzuVulsJI7FJR9hXkrKKEY4Z0tP9PltCuwg8yh68fBLuvXQi16uI4dXtdbqHsCfmZ0gBgABLtuxzvS+MtprdNN3FpBMHhS/ws7thxSnxkxG5IrtkUh+BKGMRxR0GOm4/fKahe42Xky2noLisX1w9bbjtXAc9jr3fzU4mDOpqO9ajYzvMPNy+MhUAhusLvYpEoxXOcEVm/sZmQiEAdJOOzPsMQ26wDwy3POZJW2XDD45Vw7/v7Q2R5+m2WCPy/GPzw7dj27s0pvmEsDtLzjNMSMXPmiFN+d9w6igcN7yHcY3lUTanDTtx0Kfsd7HTtSeP9JT/I1dMwiNfm4QOEqNDM17eefeOJXji6ik4dWyfrElHpnyheelIXKO8dCKAbVoQJUkun47ThVBuQjz/Jjpl4NlT465JsW5aMZfke5atCaOIzRK0BXnNt3vHdjh5TB88sWiL+8U5yJe0tLgIk4dpH0o2aRtbXzFlY1XGYnQy0/KLN7vkiHM4nrQ3By/3KOovSvE7p45yvi9CQVQIk7ZfOVZbD8J2R/JaJ8OiE8xkkQ+T8gzeM5i3stpXWswt01p33sYxMhF43dC8dAhfiVBumeGS1UL1n3H4bfO9dCLML2Y7oAiWPQtCF5aQvJATGI+XbzSLlOJ9qDd/6XCsv3WWlGcZ32SmEbf2GAWTy3u4X8ThqWumOJ5jAljKCyqkzxfJ5ms5bjrgZ82EVwpf4LNFD/rfoo2VRaz89UxcPFksdBhx2/BtXjpxuw+ayxIwKqITvwsYLdErue5y2s+ot8B8+trj8eQ1U0AIQVHKNKkoCmUsSE9UXqEffh4t1vvHNyZj4Y2nwov55vRxfYXrMyiz4ctM2oZgnjVr9kl/gwte4I/ore1lObx3R/xoxhg8fe3xvtJp365I2l0t2ZW2/OO8MpWVBHv9fv3EwzMv2dOJdiQVPj11N8UeHdth4tDumGTSaFPu8j7wswwtjEJOmv7unDG+HyYM7oZvTx+ZPVZWUoQ+ncuk3+uq38zEvZeKtyVlwcxYKU87rE/2uJVQ3DLh7EGXa9JRfviB+foJw/DE1VNwypg++NYpIzGqr/vyayfkI+wJ7KKC+9zMFrLkmnhyQ0sAwMIbT8WbP5iGAd3ah5KfGTba8NJ2vzhhQPB8JULs5iPXTNNCaZs9bxizjuiPCysG4/9mOUcH5RH0IxCWCctrObq2L8Fz3zoBQ3taNxyX1+/LSopyzLZc8yq0/sDWgF1yrHMMrTBNOlp5rJO2pr4aSk5iClbgD9c1+1SKZGfng9Koh1P2wvtzpuPFb58ode1IwcYTslg7ayZrjjCO9elchuG9O6EoBuko02EmDO4WWn5RKElRzg847eMAaMLrd+cdyZ1MZHAn6PWfMo8iCvNjPnxzWdmLOSt7rZO2Ge/dOicPN6ybpj/2jcnZczmvP9+jZeYzL98w1feLdKJTqZz939wQBnRrjwHd2uN/n261nRPdFwhTOhmOhs8IbQKbk7YXIRn15OLgHu2xZW+97/uNoXZyouzvX63g7sYW9+KzpPD6Ie/TuRTfOmUEzj3GHqhNM+kQzDqiHz7fegD9u5WxM7Zrw2ibWnwjLSECYOqo3mhXlEJTOqOiZYZFabGzcH7qminYfbDJc5qiZeFmeFqtlLbls6eKvHSMhWf2tKNw1bRN2kp5QoQHy69iaHcs2rSPW6Z8QtZue/q4vqGmHYe9OE6uOL48529CCH40Yyz3Wrbw6uqThuOy44airsl5q45wQivoXjp5EC2zYAW+iAqfrl6yL8SvVuC3bYli6bAzvDIVR6Ba84JV2a6xrcYNvxylASek3bj30mOws6YRRw/phi/+ZYHvdIJPuoqMOt74woQBePGzbYEFTxQjjBaX4fpNXxwvPG+FEO3ZdSwtRotu9xzdtzN2H3QPqJZNw1OGfLfMuClYG34UXDxZboMUsdkm/IUu5rStfZU3acvYX+99lGPJ0fFMVqiHaNL50RljcMLInjj1MGdtNwpPC95Hc8b4frj8+HIcOSjY/IMRRTU8SSDhzcnlzxcfjY2/nW1KJznpNK5/l5yYPM1p59p06+AtBIQ1PHLX9iX49zePxd843j1hPIPcNqmll9SmOm1Sw/dLjwAmHbn7wsHcRg0bvv26IHZt6bJIXSRX8yE9O+BfVx6H6poGiXzDE1ZWs1jnsuJEhaGVh6+YhM+qDuCueWsAmMoruEcURdYqiob16oiNuw95LleQJ/TSDVNz/m5O8zX8608Zie9bwk24wUw6Zo4fwd8aM7xpNcJVRpSXTgHg27shLKdfExmLsIqCnJQlWm3QeCJl+uK540f0NLKNUWHKH1GvccrYPujV2ayMMBdV95LyPowdSrTnW1ZShFW/mYlXv3uScb3PtRdBOWUMP+oqIWJPJx5eNkAR4aVPZcNdEP7xuFACP4/w7wZHHP/OZJz908+bGGyrOV6a152i+ZUX67FuowhB26WsBG/8YBpu56y+jbIDuSXtdxV3GJiX5XuwpnH55knD8cMzRuOy44airKQoG9PHnHbc9Otahmevsy+a9FOeDJtFlSAckw4rJDUd09M35xU4J3eUwI8AXiOR2YgljElbux++sw3fb37ddZspr7N997TRqLxtdlaI8/KwTtr6mTse0bsTykrsAjaoQDqTE7Nd1sX0V2cf7j/joPMM1NwG9CR9pllWUoTrp4/KEfT5ih9beElRCqWSdRO1TdnHO6JPJwzoWuZ6TxzfUmXDjwDffTeoxwYnmSg7rTmfJ6+Zkv0QAEZHlAmtEI4lK5x4QhOHdsfLy3YAAL572ihs3VePNdUHc9J00vo6+NDwTx7TG49+UInrTxnpeq2InB2d2EfeIq3OPWYgnvlEWw/iV7h4ea5hj7bCMkv+88pj5fPUe1X3DiXYV+cvtPpj39Dya2xJ4+dnjQOgjViq9tV7NkcFJf8/4a2QJFcoWjVcka0yzGBgk8p7YGQfI2yFF7MC68gVQ7ujf9eynHM/mjEGiyU2kY9iK8vvnjYad5w/wXbcKQ8/WY/s0xnv/ng6DuvfxcfdBm7x+zuXFuPOC46yX0OAW885Aq9/7yTbOX4+7tfEGR78nvnrI02/o77YcvpY7+sgrJQWF6FDO03HfuLqKbj7wqMi2U5ShNLwI0DU3KOIPy5KR9RBZfIb3rsjNuw6hEnl3fFxJX8PUK9uqDY/fP1nSVEK/buWYfsBwwvn6pOGZzcCiZOcYru4Tt576US0ZDK25zCoe3tU7eN7Qj15zRSs2l4TRlEBWGz4nPMiTVIUS8YXEcn7JCbLmdkw7BXNA7q1x5eOHoj/frzZfyI+UBp+BPj1tQ9t/0xu2v4Qf7zChZDc8n//9NGehT2LoTT7iP5Guj5Katv0WkuIy8zD++GsIwc4X8BhUnkPXDal3HO5nDA/N144Z6u8970xuksVB3Zrjz6C2D9BSMITVrRwMQzi3ixJCfwI4DXMY/VoiEcOtG/wLLrPiRxXOYGXjpfNMyqGdudcw9HQJcrnR6AQApwxzpg09bJ5d89OmlvisF6dsObmM3HhJCPyaNiLXJxekx8bfli4TdqGFTfJSZnp3bkUPzxjNN77ySmxbDIUN5EJZuWW2frhvcMZ4/thyS9OxyRB5E4v735MP5O9PMfdiy/cZD4mPAGbjcnuVyP0dC3BNdOGm8ojf+89lxyD3557BIb16oh2xalo4sS7lGfqqF74lccl/mHB/zgalQhrwtMplYHd2uP66aMiXu8Rn3R86TtT8f6c6UbeEs4Hfoj706gEfgAct19zaAjdOohX6gYPrWBPRyyoczPkCViZiV2vstHupcN89nPPeQnw1bNTqWPoCy+CQuRSK7qGHb/cEsQrLsxlHNxd2+egS3tjis5m0mFeVF4z8uKlk3dL1OQZN6ALBnRrb9pnIZq6pE2dbignGmrYKIEfgN5d+LZKv3Zvvx0kG2+buptbOpc6z9PzNiJhDZ2rP/oMhSkryGVf0jAAABT+SURBVL2YdER4MekcM0SLjeOiMEsRp83ZXNyfnzUO9182EUcPNkx0jh/uCArJthf81vRgrqZWkoxmYc773R+fItwz1wvNusCfOLS7Y3iHMAkk8Akh5xNClhNCMoSQCtPxckJIPSFkif7v3uBFzX8u0u3GvhsmR0OXYcH63QCAxZv4XjSMZ647HvN+MM3IzpIPz2ffMOnYJWDY4XfZB4RtiBL2ps4PXl4hPF9522yUc3db8leQOFelmj+OZSVFOGN87gIyq12dCZcTTOEpgmCuaqfSYlTeNjuUnczyBfPTG9yjg++Iu1ZG9NLa20WTwtntzo2gGv4yAOcCeIdzbj2l9Cj93zUB88lLrHLZ8D33J/Gj8MPPpk0IjhnSHX27GH7uV544DIN7tMeM8ZqPMV+Ld85LpOFnJw69mFP0n6w8YWn40/U4LF3ay0dVFNnw89FQwXtU5lXDU0flao+Th/XAulvOxLHDwxH4hUppcRF+9cXxOGFkNNr38SN74c0fTAsc5kSWQH74lNKVQHT2rXzHWm/R6lI/6Unfx0KucuyNIpE5vHcnvPvj6fjZ/z7XL+Yt3onv3WbD+nLcCoPw87PG4eppI4QbXYjws6lL3HTimOpOG9cXG387Cyu313K3z/S1viGhWDpAMs+9XXEKlx9fjk17vEcLlWV47+Bbm8oSpQ1/GCHkU0LI24SQqe6Xtz6s7a9J3/NWFKfD7wbnsvjpj9kPBudc1qRjzkNyEtN7OTRYwLew9twtLkpJb9jOgr6N4u0vnMc2fG55ob2bcQO6tIq4OPlMa56ANuOq4RNC5gGwR5QCbqSUPudw23YAQyilewghEwH8jxAynlJqW1pICLkKwFUAMGRIyCv+IsbaodkWiD07eVt4curYPnhjVTU4+y0LSRG+nZuzSFQIq0eGm5jPhi4IdTCqb2f7QRgfDrahRa9OcvsPOPGz2YfhQL23+Cedy0rwz28ciyNM6yX8WpbitOEfH5HJIZ9IUujm46jOD64Cn1LqHsjEfk8jgEb998WEkPUARgNYxLn2fgD3A0BFRUWCA8Zg3HXhBMwc3x+HD+iKk0ZF2/m6lGmvLUWItpmDhIQXNdj2+vLxHp1KMbZfZ6zaUWvLixecS6YP8K4Z1L091tx8Jkb/7GXutZccOxSlxUX4ckC75pVTh7tfxOFE1/fXenv/BRWD8OynW8NPOIavW6EI3SSJJJYOIaQ3gL2U0jQhZDiAUQA2RJFXvlCUSqF9O3chJeOWSSA2zTCPi6zfvX6ct8JSxsPkhtNGoW+XMpx1RH+cdlgfXPrAR/hk834AwLXTRuDdtbu5qRgTs7xz4nx5JgZW7qIUwQUReC0EFhitVh0xuP28Cbj9PHtAOFm8PoLXvndSVqFQJE9Qt8xzCCFVAKYAmEsIeVU/dRKAzwghSwE8BeAaSuneYEXNP8zyI8hqcq/xy60eMLkC3jsd2hXj6ycOQypF0KFdcY4nTx99rcG4/nwzjGMZfUWvjFaFM09segmDYH2mStOUZ3TfzhjcI5wFRbznPsbBPBg2LO+fzT4slvyiIqiXzrMAnuUcfxrA00HSbg2YJyvDDDUsn7/202p6D1oSc1WG9+qEp6+dgvEDuuI/C7eEmo8o3yjo26UM/77yWFzywEeeXD69bMjeVol78HP1tOGoqW/Gj2eMjSU/837GrRk1dR8SPA2/q6Tft9fl2+wq9pExCy9HP3yf4QUIASYO7cHdXSps4pCnE8u11ac9O0YT0bGtcUFFPP7jQG4b7tO5DL8998iso0T0eWsktcVjWCiBHwCzgLIK67svPAovXH+i/R7RQiaHtByvt/itc8viJ2qlU1oWREn7WngVg8QvLS7CH86fgP9efZz0PT84YwyKUgRDddNEW1b0rSuoLzl2aHx5SwQJjIpsX4s11/Bp3eOTPMJq0vnS0QM9p+FV4Fnz1GLp+I+WaVzrrSDi1bge0olJlPIm1kW24NPH9cX6W2dhh74xi7LhG8T5KDKZ6PN44uop3PDOxuLG6MsQJUrDD8CcMw37YSiTth67DwslbF3oleOl6aM8bqVgaTKXzaG8+DOCjJ02yEhSkL4qscVf3y6luPS4IXj4ismB8mIB2loTN31hHMYF3IYxKDKmy6BMHtYDE7n7Quj5tnIdXwl8n5x1ZH/0MXmzBBFWRigBb/ddP30UKm+bnXVxpKACG748suUY278LHrqiAr85+3DbOVGYic5l/LmNfNecCSG4+UtHYNyAYILvmetOCKlE8XHFCcPw0g1THdtXHJpvjsCPWfAWig1fmXRCQt7uLgitICnw3PIKHldfPgH3zZ29BE9LRuLfdaF/v/S2yo9mjMFlU4aicnd0MWas5OzbG7fgLRAbvtLwfVJiCTwVJO6L0YiI6X+faYXQIhNTtBPK+Jyj4/M0ae0cpZujjhzUFV3KSmIVvObNQuKX9w4eEq0MpeH75Odnjcv526sf/qg+nbC2+mDOMemFV5ZGx5ticrpWClkjvuiSCLJtLfTpXIpenUpR0+Atjk9r4KwjB+Cowd2ym5zECY3Bhu9EoXjpKA3fJz0s/r9e5P2/rzwWj18l7xboBacGGYW3jNSGVxLXsDmDQgmzvfDG0/DSDQUZIBYAcoR9nK8sV8OPV/SWFqcwY3zf0FYNJ4XS8EPCS8O3Rjb0EoxMBC+cQZSakChtL9my0VFhiPu2SRwCOJ2ght+5rAT3XSbeMa01oDT8kBjfv6v7RS5kBZ+L5LNpwhIbf8e+AMrDRyzoxjGK5Ihzoj0OP/xCR2n4Hrn7wqO4YQa6dpDfPo8xdVQvvLt2d/bvsARe1H742etCMumoUDWtlw6lWl8Y2sO+FiNscjX81m5NTwYl8D3iZwWtE49941gAwKUPfBRamk54+ZiwruQUGVCmq3npjmFvaaiIjxG9O+GBr1bguJA2Qxcx3rT+Qcl7fyiTTggM6FrmfpFHrji+HBdP5seE7+kQMMprmGU3vGz6bYXFQHdaZAUAh+krN7N9V8n7Vslp4/py99QNm16dSrHuljMjz6eQURp+QJb9agaKg8RVAHDNtBF4b93unG31bvrieACwhST+ycyx+NLRA3KOmXNftaM2Z8eqpDj3mEHYX9eMy6YMxd/eWs+95tnrjkdzOoNXlu0AkOwWdorWhVLw/aEEfkDC0GxOHNULlbfNRmNL2nbumCHdUFyUwsKN2v4xF04abHMJZfCGuVEOfUXiuShF8M2TxFsMlpUUoaykKPSRiaJwYQ4LyqTjDyXw84hifRfziyYZm7mzuCvlc+Y63he2oHTrTKFPmKlJ21D437dOQGOzXWkoJLIxbZSO7wsl8POIohTBqt/MRLsi56kVnlAU+8Pnf8dQbpnhcNTg1heF0ytOe0Ao5FACP89w21nKr1D0soqVGGpU4LRkMNwylcRXiCGE4OGvTUI5JyS3wh0l8FsZPKEokr9BYto4jQ7CNumwTdOH9mrdy9YV8XDKmD5JF6HVogR+a0OoBBuC+MOfnor99U14XPfy8aI7O31A7rnkGADAj59aql8XjkZ+ytg+eOwbk3H8iF7uFysUCt8ogd/K4MlYntbfr2sZ+nUtw5VTh2Hxpn04+6gB9htdsCrys4/sDwB4e001nlhUhZKi8EwwU0f1Di0thULBRwn8VoZXETuoewe88G37ZuriPHTXN4fzt55zBH4ycyxKi8XzDQqFIr9QAr+V4WZG+dPFR6N3J/6esfJ5iM8XF6XQM2AeCoUiflRohVaGm4b/xQkDMCWkuCbK9S0Ylx03NOkiKBQ5KIHfygjbI0eUR2vw4c9nrp42ApW3zc451reLGhkpkkMJ/FaGV7dMv7kA4X1AHrqiAm//6ORwEmvF/OD00XjvJ9OTLoaiDaNs+K2MODX8sJg+tm+4CbZSiooISgSrqBWKqFGtT2GjvKe2AIotiFKEg5oTUSRNIA2fEHIHgC8AaAKwHsDXKKX79XM/BfANAGkA36GUvhqwrArEE2/myhOHY2y/Lpg6Si2EUigKiaAa/usADqeUHglgDYCfAgAhZByAiwCMBzATwF8JIcppOwS4NvyQ80ilCE4a3Tv0mDkKhSJZAgl8SulrlNIW/c8PAQzSfz8bwOOU0kZK6UYA6wBMDpKXQkNow1deNXmN2odVkTRh2vC/DuBl/feBAMxbNVXpxxQB4e37qhTx1oGS94qkcbXhE0LmAejHOXUjpfQ5/ZobAbQA+Be7jXM9t7kTQq4CcBUADBkyhHeJwkTA3RQVCaLkvSJpXAU+pfQ00XlCyOUAzgJwKjXGrFUAzDtwDwKwzSH9+wHcDwAVFRWqT7jAs6vPGN8Pj76/CdedPDKBEikUitZCIJMOIWQmgJ8A+CKltM506nkAFxFCSgkhwwCMArAwSF4KZ7p1aIeXbpiK8l5qU4h8Rpl0FEkTdOHVXwCUAnhd1zw/pJReQyldTgh5AsAKaKaeb1FKC3uzTYXCBTWprkiaQAKfUupoQ6CU3gLgliDpKxSFhNLwFUmjQiso2izPfesEdCyNb3mIkveKpFGhFRRtlgmDu2Fkn86R53PexEHuFykUMaAEfithRO/WPyH7nekj0a8NxucZ3F3fnF3ZdBQJo0w6rYRnrjsBu2obki5GIL5/xhh8/4wxSRcjdoz9BRSKZFECv5XQtX0JurYvSboYigAoBV+RNMqko1BEDFsqp9wyFUmjBL5CETFZk46S94qEUQJfoYgYFWZakS8oga9QxIRS8BVJowS+QhExyqSjyBeUwFcoIqadvnF5SZEy7SiSRbllKhQRc+lxQ7GrthHXnjwi6aIo2jhK4CsUEVNWUoSfzjos6WIoFMqko1AoFG0FJfAVCoWijaAEvkKhULQRlMBXKBSKNoIS+AqFQtFGUAJfoVAo2ghK4CsUCkUbQQl8hUKhaCMQmkcBPgghtQBW+7x9CIDNIRbHja4ADsSUVyHXDSjs+hVy3YB461fIdQOC1W8MpdR1g+Z8E/iLKKUVPu/dRSntHXaZBPndTym9Kqa8CrZuen4FW79CrpueX2z1K+S66fn5rp+s7Cwkk87+mPN7Ica8CrluQGHXr5DrBsRbv0KuGxBD/QpJ4Mc51AOlNM7GV8h1Awq7foVcNyDG+hVy3YB46pdvAv/+hO7Ndwq5bkBh16+Q6wYUdv1aU92kyppXNnyFQqFQREe+afgKhUKhiIi8FfiEkMGEkPmEkJWEkOWEkBv04z0IIa8TQtbqP7vrxwkh5E+EkHWEkM8IIcfox48ihHygp/EZIeTCJOullymUupnS60II2UoI+UsS9bESZv0IIUMIIa/paa0ghJQnU6tsecKs2+16Giv1axLfEstH/cbq/auREPJDS1ozCSGr9brPSaI+lvKEUjendFoFlNK8/AegP4Bj9N87A1gDYByA2wHM0Y/PAfA7/fdZAF4GQAAcB+Aj/fhoAKP03wcA2A6gWyHUzZTeHwH8G8Bfkn5vYdcPwFsATtd/7wSgQyHUDcDxABYAKNL/fQDg5Fb47voAmATgFgA/NKVTBGA9gOEA2gFYCmBcgdSNm07S707mX95q+JTS7ZTST/TfawGsBDAQwNkAHtUvexTAl/TfzwbwD6rxIYBuhJD+lNI1lNK1ejrbAFQDiM23lkdYdQMAQshEAH0BvBZjFYSEVT9CyDgAxZTS1/W0DlJK6+Ksi5UQ3x0FUAZNGJYCKAGwM7aKOOC1fpTSakrpxwCaLUlNBrCOUrqBUtoE4HE9jcQIq26CdPKevBX4ZvRh/NEAPgLQl1K6HdAePLSvMKA98C2m26pgeQmEkMnQOtj6aEssT5C6EUJSAP4A4EdxldcrAd/daAD7CSHPEEI+JYTcQQgpiqvsbgSpG6X0AwDzoY04twN4lVK6Mp6SyyFZPydc+2OSBKybUzp5T94LfEJIJwBPA/gupbRGdCnnWNYFSdeqHgPwNUppJtxS+iOEul0H4CVK6RbO+cQJoX7FAKYC+CG0ofVwAFeEXExfBK0bIWQkgMMADIImCKcTQk4Kv6T+8FA/xyQ4x/LCJTCEuoWaTpzktcAnhJRAe6D/opQ+ox/eaTJn9IdmogE0DWKw6fZBALbp13UBMBfAz/RhdeKEVLcpAK4nhFQC+D2ArxJCbouh+K6EVL8qAJ/qZoEWAP8DkDNhnQQh1e0cAB/qZqqD0Oz8x8VRfjc81s8Jx/6YJCHVzSmdvCdvBb7usfAggJWU0jtNp54HcLn+++UAnjMd/6ruFXEcgAOU0u2EkHYAnoVmR30ypuILCatulNKvUEqHUErLoWnB/6CU5oM3RCj1A/AxgO6EEDbnMh3AisgrICDEum0GMI0QUqwLj2nQbMGJ4qN+TnwMYBQhZJjeBy/S00iMsOomSCf/iXpW2O8/ACdCGwJ+BmCJ/m8WgJ4A3gCwVv/ZQ7+eALgHmn3+cwAV+vFLoU26LDH9O6oQ6mZJ8wrkj5dOaPUDcLqezucAHgHQrhDqBs2L5T5oQn4FgDuTfm8+69cPmjZfAy32TBWALvq5WdA8WNYDuLFQ6uaUTtL1k/mnVtoqFApFGyFvTToKhUKhCBcl8BUKhaKNoAS+QqFQtBGUwFcoFIo2ghL4CoVC0UZQAl/RpiGEpAkhS/Soh0sJId/XQ1aI7iknhFwSVxkVirBQAl/R1qmnlB5FKR0Pzed/FoBfutxTDkAJfEWrQ/nhK9o0hJCDlNJOpr+HQ1sl2gvAUGjxlzrqp6+nlL5PCPkQWhycjdCiK/4JwG0AToYW+fIeSul9sVVCoZBECXxFm8Yq8PVj+wCMBVALIEMpbSCEjALwH0ppBSHkZGjx0c/Sr78KQB9K6c2EkFJoce7Pp5RujLUyCoULxUkXQKHIQ1ikxxIAfyGEHAUgDS1cM48zABxJCDlP/7srgFHQRgAKRd6gBL5CYUI36aShRUz8JbRNSSZAm+9qcLoNwLcppa/GUkiFwidq0lah0NGjct4LLQgdhaapb6fa/gmXQQt4Bmimns6mW18FcK0e9RKEkNGEkI5QKPIMpeEr2jrtCSFLoJlvWqBN0rKQt38F8DQh5Hxou1Md0o9/BqCFELIUWgTPP0Lz3PlED527C8YWhwpF3qAmbRUKhaKNoEw6CoVC0UZQAl+hUCjaCErgKxQKRRtBCXyFQqFoIyiBr1AoFG0EJfAVCoWijaAEvkKhULQRlMBXKBSKNsL/A47qdE6f244nAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%matplotlib inline \n",
    "import matplotlib, matplotlib.pyplot as plt\n",
    "daily_test=obs[\"Pyramid\"].resample(\"D\").min()\n",
    "#print(daily_test.head(3))\n",
    "# Plots\n",
    "daily_test[\"Temp\"].plot(); print (np.min(daily_test[\"Temp\"]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
