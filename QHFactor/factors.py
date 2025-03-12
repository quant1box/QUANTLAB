import pandas as pd
import polars as pl
from QHFactor.fn import *


# -------------------------------------
# 國泰君安191因子
# -------------------------------------
class GTJA_191:

    def __init__(self, dataset: pl.DataFrame) -> None:
        """
        初始化因子计算器
        """
        if isinstance(dataset, pd.DataFrame):
            dataset = pl.from_pandas(dataset)
        else:
            dataset = dataset.clone()

        self.open = dataset['open'].to_numpy()
        self.high = dataset['high'].to_numpy()
        self.close = dataset['close'].to_numpy()
        self.low = dataset['low'].to_numpy()
        self.volume = dataset['volume'].to_numpy()

        self.returns = returns(self.open, self.close)
        self.vwap = vwap(self.close, self.volume, 20)

        self.HD = self.high - delay(self.high, 1)
        self.LD = delay(self.low, 1) - self.low
        self.TR = maximum(maximum(self.high - self.low, abs(self.high -
                          delay(self.close, 1))), abs(self.low - delay(self.close, 1)))

    def alpha_001(self,  n: int = 6) -> np.ndarray:
        """
        -1 *CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6)
        """
        data = -1*ts_corr(rank(delta(log(self.volume), 1)),
                          rank(((self.close - self.open) / self.open)), n)

        return data

    def alpha_002(self,  n: int = 1) -> np.ndarray:
        """
        -1 *DELTA((((CLOSE - LOW)- (HIGH- CLOSE))/(HIGH-LOW)),1)
        """
        data = -1 * delta((((self.close - self.low) -
                            (self.high - self.close)) / (self.high - self.low)), n)

        return data

    def alpha_003(self,  n: int = 6) -> np.ndarray:
        """
        SUM((CLOSE=DELAY(CLOSE, 1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1))),6)
        """
        data = ts_sum(np.where((self.close == delay(self.close, 1)), 0,
                               (self.close - np.where((self.close > delay(self.close, 1)),
                                                      minimum(self.low, delay(
                                                          self.close, 1)),
                                                      maximum(self.high, delay(self.close, 1))))), n)

        return data
    
    def alpha_004(self, n: int = 6) -> np.ndarray:
        """
        (((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1):(((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8)) ? 1: ((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME / MEAN(VOLUME,20))==1))?1:(-1*1)))
        """
        close_sum_8 = ts_sum(self.close, 8)
        close_std_8 = ts_stddev(self.close, 8)
        close_sum_2 = ts_sum(self.close, 2)
        volume_mean_20 = ts_mean(self.volume, 20)

        cond1 = (close_sum_8 / 8 + close_std_8) < (close_sum_2 / 2)
        cond2 = (close_sum_2 / 2) < (close_sum_8 / 8 - close_std_8)
        cond3 = 1 <= (self.volume / volume_mean_20)

        data = np.where(cond1, -1, np.where(cond2, 1, np.where(cond3, 1, -1)))
        return data

    def alpha_005(self,  n: int = 3) -> np.ndarray:
        """
        (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK (HIGH, 5), 5), 3))
        """

        volume_rank = ts_rank(self.volume, 5)
        high_rank = ts_rank(self.high, 5)
        correlation = ts_corr(volume_rank, high_rank, 5)

        data = -1 * ts_max(correlation, n)
        return data

    def alpha_006(self,  n: int = 4) -> np.ndarray:
        """
        (RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)
        (RANK/SIGN(DELTA(((IOPEN * 0.85) +(HIGHI*0.15))),4)))* 1)
        """
        data = rank(
            sign(delta((((self.open * 0.85) + (self.high * 0.15))), n))) * -1

        return data

    def alpha_007(self,  n: int = 3) -> np.ndarray:
        """
        (RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * \
         RANK(DELTA(VOLUME, 3))
        """

        data = ((rank(ts_max((self.vwap - self.close), n)) +
                rank(ts_min((self.vwap - self.close), n))) * rank(delta(self.volume, n)))

        return data

    def alpha_008(self,  n: int = 4) -> np.ndarray:
        """
        RANK(DELTA(((((HIGH + LOW) /2) *0.2)+ (VWAP *0.8),4)*-1)
        """
        data = rank(delta((((self.high + self.low)/2) * 0.2) +
                    (self.vwap * 0.8), n) * -1)

        return data

    def alpha_009(self,  n: int = 7) -> np.ndarray:
        """
        SMA((HIGH+LOW)/2-(DELAY(HIGH, 1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME, 7,2)
        """
        data = ts_sma(((self.high + self.low)/2 - (delay(self.high, 1) +
                       delay(self.low, 1))/2) * (self.high - self.low)/self.volume, n, 2)

        return data

    def alpha_010(self,  n: int = 5) -> np.ndarray:
        """
        (RANK(MAX(((RET < O) ? STD(RET, 20) :CLOSE)^2),5))
        """
        data = rank(
            ts_max(np.where((self.returns < 0), ts_stddev(self.returns, 20), self.close)**2, n))

        return data

    def alpha_011(self,  n: int = 6) -> np.ndarray:
        """

        SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)
        """
        data = ts_sum(((self.close - self.low) - (self.high - self.close)) /
                      (self.high - self.low)*self.volume, n)

        return data

    def alpha_012(self,  n: int = 10) -> np.ndarray:
        """
        (RANK((OPEN - (SUM(VWAP, 10) /10)))) *(-1 * (RANK(ABS/(CLOSE - VWAP)))))
        """
        data = (rank((self.open - ts_sum(self.vwap, n)/n)) *
                (-1 * (rank(abs(self.close - self.vwap)))))

        return data

    def alpha_013(self,  n: int = 20) -> np.ndarray:
        """
        (((HIGH * LOW)^0.5) - VWAP)
        """
        data = (mul(self.high, self.low)**0.5 - self.vwap)
        return data

    def alpha_014(self,  n: int = 5) -> np.ndarray:
        """
        CLOSE-DELAY(CLOSE,5)
        """
        data = self.close - delay(self.close, n)

        return data

    def alpha_015(self,  n: int = 1) -> np.ndarray:
        """
        OPEN/DELAY(CLOSE,1)-1
        """
        data = self.open/delay(self.close, 1) - 1

        return data

    def alpha_016(self,  n: int = 5) -> np.ndarray:
        """
        (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))
        """
        data = -1 * \
            ts_max(rank(ts_corr(rank(self.volume), rank(self.vwap), n)), n)

        return data

    def alpha_017(self,  n: int = 5) -> np.ndarray:
        """
        RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)
        """
        data = rank((self.vwap - ts_max(self.vwap, 15)))*delta(self.close, n)
        return data

    def alpha_018(self,  n: int = 5) -> np.ndarray:
        """
        CLOSE/DELAY(CLOSE,5)
        """

        data = self.close/delay(self.close, n)

        return data

    def alpha_019(self,  n: int = 5) -> np.ndarray:
        """
        (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-D
        ELAY(CLOSE,5))/CLOSE))
        """

        data = np.where((self.close < delay(self.close, n)),
                        (self.close - delay(self.close, n))/delay(self.close, n),
                        np.where((self.close == delay(self.close, n)), 0,
                                 (self.close - delay(self.close, n)/self.close)))

        return data

    def alpha_020(self,  n: int = 6) -> np.ndarray:
        """
        (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
        """

        data = (self.close - delay(self.close, n))/delay(self.close, n)*100

        return data

    def alpha_021(self, n: int = 6) -> np.ndarray:
        """
        REGBETA(MEAN(CLOSE, 6),SEQUENCE(6))
        """
        # TOD
        means = ts_mean(self.close, n)
        regbeta = np.empty_like(means)

        for i in range(n-1, len(means)):
            m_n = means[i-n+1: i+1]
            regbeta[i] = reg_beta(m_n, sequence(n))

        return reg_beta

    def alpha_022(self,  n: int = 6) -> np.ndarray:
        """
        SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
        """
        temp = (self.close - ts_mean(self.close, 6))/ts_mean(self.close, 6)
        data = ts_sma((temp - delay(temp, 3)), 12, 1)

        return data

    def alpha_023(self,  n: int = 20) -> np.ndarray:
        """
        SMA((CLOSE>DELAY(CLOSE, 1) ?STD(CLOSE:20),0),20.1)/(SMA(CLOSE>DELAY/CLOSE, 1) ?STD(CLOSE,20):0).20.1 )+SMA((CLOSE<=DELAY(CLOSE, 1)?STD(CLOSE,20):0),20,1))*100
        """
        value1 = ts_sma(
            np.where((self.close > delay(self.close, 1)), ts_stddev(self.close, n), 0), n, 1)
        value2 = ts_sma(
            np.where((self.close <= delay(self.close, 1)), ts_stddev(self.close, n), 0), n, 1)
        data = value1 / (value1 + value2) * 100

        return data

    def alpha_024(self,  n: int = 5) -> np.ndarray:
        """
        SMA(CLOSE-DELAY(CLOSE,5),5,1)
        """
        data = ts_sma((self.close - delay(self.close, n)), n, 1)

        return data

    def alpha_025(self, n: int = 5) -> np.ndarray:
        """
        ((-1*RANK((DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9))))))*(1+RANK(SUM(RET,250))))
        """
        data = ((-1*rank((delta(self.close, 7)*(1 - rank(decay_linear((self.volume /
                ts_mean(self.volume, 20)), 9))))))*(1 + rank(ts_sum(self.returns, 20))))

        return data

    def alpha_026(self, n: int = 20) -> np.ndarray:
        """
        ((((SUM(CLOSE,7)/7)-CLOSE))+((CORR(VWAP,DELAY(CLOSE,5),230))))
        """
        data = (ts_sum(self.close, 7)/7 - self.close) + \
            ts_corr(self.vwap, delay(self.close, 5), 20)

        return data

    def alpha_027(self, n: int = 12) -> np.ndarray:
        """
        WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
        """
        data = ts_wma(((self.close - delay(self.close, 3))/delay(self.close, 3)
                      * 100 + (self.close - delay(self.close, 6))/delay(self.close, 6)*100), n)
        return data

    def alpha_028(self,  n: int = 3) -> np.ndarray:
        """
        3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
        """

        data = 3 * ts_sma((self.close - ts_min(self.low, n))/(ts_max(self.high, n) - ts_min(self.low, n))*100, n, 1) - 2 * ts_sma(ts_sma((self.close - ts_min(self.low, n)) /
                                                                                                                                         (ts_max(self.high, n) - ts_max(self.low, n))*100, n, 1), n, 1)

        return data

    def alpha_029(self,  n: int = 6) -> np.ndarray:
        """
        (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
        """

        data = (self.close - delay(self.close, n)) / \
            delay(self.close, n) * self.volume

        return data

    def alpha_030(self, n: int = 10) -> np.ndarray:
        """
        WMA((REGRESI/CLOSE/DELAY(CLOSE)-1,MKT,SMB, HNL,60))^2,20)
        """
        # TODO
        pass

    def alpha_031(self,  n: int = 12) -> np.ndarray:
        """
        (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
        """
        data = (self.close - ts_mean(self.close, n))/ts_mean(self.close, n)*100

        return data

    def alpha_032(self,  n: int = 3) -> np.ndarray:
        """
        (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
        """
        data = -1 * \
            ts_sum(rank(ts_corr(rank(self.high), rank(self.volume), n)), n)

        return data

    def alpha_033(self,  n: int = 5) -> np.ndarray:
        """
        ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *
        TSRANK(VOLUME, 5))
        """
        data = ((((-1 * ts_min(self.low, n)) + delay(ts_min(self.low, n), n)) * rank(((ts_sum(self.returns, 20) - ts_sum(self.returns, 5)) / 20))) *
                ts_rank(self.volume, n))

        return data

    def alpha_034(self,  n: int = 12) -> np.ndarray:
        """
        MEAN(CLOSE,12)/CLOSE
        """

        data = ts_mean(self.close, n)/self.close
        return data

    def alpha_035(self, n: int = 20) -> np.ndarray:
        """
        (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65)+ (OPEN *0.35)),17),7))))* -1)
        """
        data = (minimum(rank(decay_linear(delta(self.open, 1), 15), rank(delay(ts_corr((self.volume), ((self.open * 0.65) + (self.open * 0.35)), 17),7))))*-1)
        return data

    def alpha_036(self, n: int = 15) -> np.ndarray:
        """
        (MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)),RANK(DECAYLINEAR(CORR((VOLUME),((OPEN*0.65)+(OPEN*0.35)),17),7)))*-1)
        """
        data = (minimum(rank(decay_linear(delta(self.open, 1), 15)), rank(decay_linear(
            ts_corr((self.volume), ((self.open * 0.65)+(self.open*0.35)), 17), 7))) * -1)

        return data

    def alpha_037(self, n: int = 5) -> np.ndarray:
        """
        (-1*RANK(((SUM(OPEN,5)*SUM(RET,5))-DELAY((SUM(OPEN,5)*SUM(RET,5)),10))))
        """

        data = (-1*rank(((ts_sum(self.open, n) * ts_sum(self.returns, n)) -
                delay((ts_sum(self.open, n) * ts_sum(self.returns, n)), 10))))

        return data

    def alpha_038(self, n: int = 2) -> np.ndarray:
        """
        (((SUM(HIGH,20)/20)<HIGH)?(-1*DELTA(HIGH,2)):0)
        """
        data = np.where(((ts_sum(self.high, 20)/20) < self.high),
                        (-1 * delta(self.high, n)), 0)

        return data

    def alpha_039(self, n: int = 5) -> np.ndarray:
        """
        ((RANK(DECAYLINEAR(DELTA((CLOSE),2),8))-RANK(DECAYLINEAR(CORR(((VWAP*0.3)+(OPEN*0.7)),SUM(MEAN(VOLUME,180),37),14),12)))*-1
        """
        data = (rank(decay_linear(delta((self.close), 2), 8)) - rank(decay_linear(ts_corr(
            ((self.vwap*0.3)+(self.open*0.7)), ts_sum(ts_mean(self.volume, 20), 6), 5), 4))) * -1

        return data

    
    def alpha_040(self, n: int = 26) -> np.ndarray:
        """
        SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
        """
        # Ensure that the condition is correctly applied and the denominator is not zero
        volume_sum_positive = ts_sum(np.where((self.close > delay(self.close, 1)), self.volume, 0), n)
        volume_sum_negative = ts_sum(np.where(self.close <= delay(self.close, 1), self.volume, 0), n)
        
        # Avoid division by zero
        data = np.where(volume_sum_negative != 0, volume_sum_positive / volume_sum_negative * 100, 0)
        return data

    def alpha_041(self, n: int = 5) -> np.ndarray:
        """
        (RANK(MAX(DELTA((VWAP),3),5))*-1)
        """
        data = rank(ts_max(delta(self.vwap, 3), n)) * -1

        return data

    def alpha_042(self, n: int = 10) -> np.ndarray:
        """
        (-1*RANK(STD(HIGH,10)))*CORR(HIGH,VOLUME,10))
        """

        data = -1 * rank(ts_stddev(self.high, n)) * \
            ts_corr(self.high, self.volume, n)
        return data

    def alpha_043(self, n: int = 6) -> np.ndarray:
        """
        SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
        """

        data = ts_sum(np.where((self.close > delay(self.close, 1)),
                               self.volume, np.where((self.close < delay(self.close, 1)),
                                                     -1*self.volume, 0)), n)

        return data

    def alpha_044(self, n: int = 15) -> np.ndarray:
        """
        (TSRANK(DECAYLINEAR(CORR(((LOW)),MEAN(VOLUME,10),7),6),4)+TSRANK(DECAYLINEAR(DELTA((VWAP),3),10),15))
        """

        data = (ts_rank(decay_linear(ts_corr(((self.low)), ts_mean(
            self.volume, 10), 7), 6), 4) + ts_rank(decay_linear(delta((self.vwap), 3), 10), n))

        return data

    def alpha_045(self, n: int = 5) -> np.ndarray:
        """
        (RANK(DELTA((((CLOSE*0.6)+(OPEN*0.4))),1))*RANK(CORR(VWAP,MEAN(VOLUME,150),15)))
        """

        data = (rank(delta((((self.close*0.6)+(self.open*0.4))), 1)) *
                rank(ts_corr(self.vwap, ts_mean(self.volume, 20), n)))
        return data

    def alpha_046(self, n: int = 3) -> np.ndarray:
        """
        (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
        """

        data = (ts_mean(self.close, n) + ts_mean(self.close, 2*n) +
                ts_mean(self.close, 4*n) + ts_mean(self.close, 8*n))/(4*self.close)

        return data

    def alpha_047(self, n: int = 6) -> np.ndarray:
        """
        SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
        """

        data = ts_sma((ts_max(self.high, n) - self.close) /
                      (ts_max(self.high, n) - ts_min(self.low, n))*100, 9, 1)

        return data

    def alpha_048(self, n: int = 5) -> np.ndarray:
        """
        (-1*((RANK(((SIGN((CLOSE-DELAY(CLOSE,1)))+SIGN((DELAY(CLOSE,1)-DELAY(CLOSE,2))))+SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3))))))*SUM(VOLUME,5))/SUM(VOLUME,20))
        """
        data = (-1*((rank(((sign((self.close-delay(self.close, 1)))+sign((delay(self.close, 1)-delay(self.close, 2)))) +
                sign((delay(self.close, 2)-delay(self.close, 3))))))*ts_sum(self.volume, n))/ts_sum(self.volume, 4*n))

        return data

    def alpha_049(self, n: int = 12) -> np.ndarray:
        """
        SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
        """
        con1 = (self.high + self.low) >= (delay(self.high, 1) + delay(self.low, 1))
        con2 = (self.high + self.low) <= (delay(self.high, 1) + delay(self.low, 1))

        value1 = maximum(abs(self.high - delay(self.high)),
                         abs(self.low - delay(self.low, 1)))

        data = ts_sum(np.where(con1, 0, value1), n)/(ts_sum(np.where(con1,
                                                                     0, value1), n) + ts_sum(np.where(con2, 0, value1), n))

        return data

    def alpha_050(self, n: int = 6) -> np.ndarray:
        """
        SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0: MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELA Y(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
        """

        pass

    def alpha_051(self, n: int = 6) -> np.ndarray:
        """
        SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
        """
        pass

    def alpha_052(self, n: int = 26) -> np.ndarray:
        """
        SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100
        """
        data = ts_sum(maximum(0, self.high - delay((self.high + self.low + self.close)/3, 1)), 26) / \
            ts_sum(
            maximum(0, delay((self.high+self.low+self.close)/3, 1) - self.low), n)*100
        return data

    def alpha_053(self, n: int = 12) -> np.ndarray:
        """
        COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
        """

        data = count(self.close > delay(self.close, 1), n)/n*100
        return data

    def alpha_054(self, n: int = 10) -> np.ndarray:
        """
        (-1*RANK((STD(ABS(CLOSE-OPEN))+(CLOSE-OPEN))+CORR(CLOSE,OPEN,10)))
        """
        data = (-1*rank((ts_stddev(abs(self.close - self.open), n)) +
                (self.close - self.open)) + ts_corr(self.close, self.open, n))

        return data

    def alpha_055(self, n: int = 20) -> np.ndarray:
        """
        SUM(16*(CLOSE-DEL.AY(CLOSE, I)+(CLOSE-OPEN)/2+DELAY(CLOSE, 1)-DEL.AY(OPEN.I))/(ABS(HIGHI-DEL.AY(CL OSE, I))>ABS(LOW-DELAY(CLOSE, 1)) & ABS(HIGH-DELAY(CLOSE, I))>ABS(HIGH-DELAY(LOW.L))?ABS(HIGH-DELAY(CLOSE, I)) +ABS(LOW-DELAY(CLOS E,L))/2+ABS(DELAY(CLOSE, I)-DELAY(OPEN,L))/4:(ABS(LOW-DELAY(CLOSE,I))>ABS(HIGH-DELAY(LOW.L))& ABS(LOW-DEL.AY(CLOSE, I))>ABS(HIGH-DELAY(CLOSE, 1)) ?ABS(LOW-DEL.AY(CLOSE, 1)) +ABS(HIGHI-DEL.AY(CLO SE, 1))/2+ABS(DELAY(CLOSE, 1)-DELAY(OPEN, 1))/4:ABS(HIGH-DELAY(LOW.I)) +ABS(DELAY(CLOSE, 1)-DELAY(OP EN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE, I)),ABS(LOW-DELAY(CLOSE,I))),20)
        """
        pass

    def alpha_056(self, n: int = 10) -> np.ndarray:
        """
        (RANK((OPEN-TSMIN(OPEN, 12))) <
         RANK((RANK(CORR(SUM(((HIIGH+ LOW) /2), 19),SUM(MEAN(VOLUME, 40), 19), 13))^5)))
        """
        data = rank((self.open - ts_min(self.open, 12))) < rank((rank(ts_corr(ts_sum(
            ((self.high + self.low) / 2), 19)/ts_sum(ts_mean(self.volume, 40), 19), 13))**5))
        return data

    def alpha_057(self, n: int = 9) -> np.ndarray:
        """
        SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
        """
        data = ts_sma((self.close - ts_min(self.low, n)) /
                      (ts_max(self.high, n)-ts_min(self.low, n))*100, 3, 1)

        return data

    def alpha_058(self, n: int = 20) -> np.ndarray:
        """
        COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
        """

        data = count(self.close > delay(self.close, 1), n)/n*100
        return data

    def alpha_059(self, n: int = 20) -> np.ndarray:
        """
        SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)
        """
        cond1 = self.close == delay(self.close, 1)
        cond2 = self.close > delay(self.close, 1)
        minest = minimum(self.low, delay(self.close, 1))
        maxest = maximum(self.high, delay(self.close, 1))

        data = ts_sum(
            np.where(cond1, 0, (self.close - np.where(cond2, minest, maxest))), n)

        return data

    def alpha_060(self, n: int = 20) -> np.ndarray:
        """
        SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)
        """

        data = ts_sum(((self.close - self.low) - (self.high -
                                                  self.close))/(self.high - self.low)*self.volume, n)
        return data

    def alpha_061(self, n: int = 17) -> np.ndarray:
        """
        (MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80),8)),17)))*-1)
        """
        data = (maximum(rank(decay_linear(delta(self.vwap, 1), 12)), rank(
            decay_linear(rank(ts_corr((self.close), ts_mean(self.volume, 40), 4)), n)))*-1)

        return data

    def alpha_062(self, n: int = 5) -> np.ndarray:
        """
        (-1*CORR(HIGH,RANK(VOLUME),5))
        """

        data = (-1*ts_corr(self.high, rank(self.volume), n))

        return data

    def alpha_063(self, n: int = 6) -> np.ndarray:
        """
        SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
        """

        data = ts_sma(maximum(self.close - delay(self.close, 1), 0), n, 1) / \
            ts_sma(abs(self.close - delay(self.close, 1)), n, 1)*100

        return data

    def alpha_064(self, n: int = 6) -> np.ndarray:
        """
        (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE),RANK(MEAN(VOLUME,60)),4),13),14)))*-1)
        """
        data = (maximum(rank(decay_linear(ts_corr(rank(self.vwap), rank(self.volume), 4), 4)), rank(
            decay_linear(maximum(ts_corr(rank(self.close), rank(ts_mean(self.volume, 60)), 4), 13), 14)))*-1)
        return data

    def alpha_065(self, n: int = 6) -> np.ndarray:
        """
        MEAN(CLOSE,6)/CLOSE
        """
        data = ts_mean(self.close, n)/self.close
        return data

    def alpha_066(self, n: int = 6) -> np.ndarray:
        """
        (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
        """

        data = (self.close - ts_mean(self.close, n))/ts_mean(self.close, n)*100

        return data

    def alpha_067(self, n: int = 24) -> np.ndarray:
        """
        SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
        """
        data = ts_sma(maximum(self.close - delay(self.close, 1), 0), n, 1) / \
            ts_sma(abs(self.close - delay(self.close, 1)), n, 1)*100

        return data

    def alpha_068(self, n: int = 15) -> np.ndarray:
        """
        SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
        """

        data = ts_sma(((self.high + self.low)/2 - (delay(self.high, 1) +
                                                   delay(self.low, 1))/2)*(self.high - self.low)/self.volume, n, 2)
        return data

    def alpha_069(self, n: int = 15) -> np.ndarray:
        """
        DTM = (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
        DBM = (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))

        (SUM(DTM,20)>SUM(DBM,20)?(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):(SUM(DTM,20)=SUM(DBM,20)？0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
        """
        DTM = np.where((self.open <= delay(self.open, 1)), 0, maximum(
            (self.high - self.open), (self.open - delay(self.open, 1))))

        DBM = np.where((self.open >= delay(self.open, 1)), 0, maximum(
            (self.open - self.low), (self.open - delay(self.open, 1))))

        SUM_DTM = ts_sum(DTM, n)
        SUM_DBM = ts_sum(DBM, n)

        data = np.where((SUM_DTM > SUM_DBM), ((SUM_DTM - SUM_DBM)/SUM_DTM),
                        np.where((SUM_DTM == SUM_DBM), 0, ((SUM_DTM - SUM_DBM)/SUM_DBM)))

        return data

    def alpha_070(self, n: int = 6) -> np.ndarray:
        """
        STD(AMOUNT,6)
        """

        data = ts_stddev(self.close*self.volume, n)
        return data

    def alpha_071(self, n: int = 24) -> np.ndarray:
        """
        (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
        """
        data = (self.close - ts_mean(self.close, n))/ts_mean(self.close, n)*100

        return data

    def alpha_072(self, n: int = 15) -> np.ndarray:
        """
        SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
        """
        data = ts_sma((ts_max(self.high, 6) - self.close) /
                      (ts_max(self.high, 6) - ts_min(self.low, 6)), n, 1)
        return data

    def alpha_073(self, n: int = 3) -> np.ndarray:
        """
        ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE),VOLUME,10),16),4),5)-RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3)))*-1)
        """
        data = ((ts_rank(decay_linear(decay_linear(ts_corr((self.close), self.volume, 10), 16), 4),
                         5) - rank(decay_linear(ts_corr(self.vwap, ts_mean(self.volume, 30), 4), 3)))*-1)
        return data

    def alpha_074(self, n: int = 6) -> np.ndarray:
        """
        (RANK(CORR(SUM(((LOW*0.35)+(VWAP*0.65)),20),SUM(MEAN(VOLUME,40),20),7))+RANK(CORR(RANK(VWAP),RANK(VOLUME),6)))
        """

        data = (rank(ts_corr(ts_sum(((self.low*0.35)+(self.vwap*0.65)), 20), ts_sum(ts_mean(
            self.volume, 40), 20), 7))+rank(ts_corr(rank(self.vwap), rank(self.volume), n)))
        return data

    def alpha_075(self, n: int = 6) -> np.ndarray:
        """
        BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKIN DEXOPEN,50)
        """
        # TODO
        pass

    def alpha_076(self, n: int = 20) -> np.ndarray:
        """
        STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
        """
        temp = np.abs((self.close/delay(self.close, 1) - 1))/self.volume
        data = ts_stddev(temp, n) / ts_mean(temp, n)

        return data

    def alpha_077(self, n: int = 6) -> np.ndarray:
        """
        MIN(RANK(DECAYLINEAR(((((HIGH+LOW)/2)+HIGH)-(VWAP+HIGH)),20)),RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),3),6)))

        """
        data = minimum(rank(decay_linear(((((self.high + self.low)/2) + self.high) - (self.vwap + self.high)), 20)),
                       rank(decay_linear(ts_corr(((self.high + self.low)/2), ts_mean(self.volume, 40), 3), n)))

        return data

    def alpha_078(self, n: int = 12) -> np.ndarray:
        """
        ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
        """
        triple = (self.high + self.low + self.close)/3
        data = (triple - ts_mean(triple, n)) / \
            (0.015*ts_mean(abs(self.close - ts_mean(triple, n)), n))

        return data

    def alpha_079(self, n: int = 12) -> np.ndarray:
        """
        SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
        """

        data = ts_sma(maximum(self.close - delay(self.close, 1), 0), n, 1) / \
            ts_sma(abs(self.close - delay(self.close, 1)), n, 1)*100
        return data

    def alpha_080(self, n: int = 5) -> np.ndarray:
        """
        (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
        """

        data = (self.volume - delay(self.volume, n))/delay(self.volume, n)*100

        return data

    def alpha_081(self, n: int = 21) -> np.ndarray:
        """
        SMA(VOLUME,21,2)
        """

        data = ts_sma(self.volume, n, 2)

        return data

    def alpha_082(self, n: int = 20) -> np.ndarray:
        """
        SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
        """
        data = ts_sma((ts_max(self.high, 6)-self.close) /
                      (ts_max(self.high, 6) - ts_min(self.low, 6)*100), n, 1)

        return data

    def alpha_083(self, n: int = 5) -> np.ndarray:
        """
        (-1*RANK(COVIANCE(RANK(HIGH),RANK(VOLUME),5)))
        """
        data = (-1*rank(ts_cov(rank(self.high), rank(self.volume), n)))

        return data

    def alpha_084(self, n: int = 20) -> np.ndarray:
        """
        SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
        """

        data = ts_sum(np.where((self.close > delay(self.close)), self.volume,
                               np.where((self.close < delay(self.close, 1)), -self.volume, 0)), n)

        return data

    def alpha_085(self, n: int = 20) -> np.ndarray:
        """
        (TSRANK((VOLUME/MEAN(VOLUME,20)),20)*TSRANK((-1*DELTA(CLOSE,7)),8))
        """
        data = (ts_rank((self.volume/ts_mean(self.volume, n)), n)
                * ts_rank((-1*delta(self.close, 7)), 8))
        return data

    def alpha_086(self, n: int = 20) -> np.ndarray:
        """
        ((0.25<(((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)-((DELAY(CLOSE,10)-CLOSE)/10)))?(-1*1):(((((DELAY(CLOSE,20)-DELAY(CLOSE,10))/10)-((DELAY(CLOSE,10)-CLOSE)/10))\<0)?1:((-1*1)*(CLOSE-DELAY(CLOSE,1)))))
        """
        v1 = (((delay(self.close, 20) - delay(self.close, 10))/10) -
              ((delay(self.close, 10) - self.close)/10))

        data = np.where((0.25 < v1), -1, np.where((v1 < 0),
                                                  1, -1*(self.close - delay(self.close, 1))))
        return data

    def alpha_087(self, n: int = 20) -> np.ndarray:
        """
        ((RANK(DECAYLINEAR(DELTA(VWAP,4),7))+TSRANK(DECAYLINEAR(((((LOW*0.9)+(LOW*0.1))-VWAP)/(OPEN-((HIGH+LOW)/2))),11),7))*-1)
        """
        data = ((rank(decay_linear(delta(self.vwap, 4), 7)) + ts_rank(decay_linear(((((self.low*0.9) +
                                                                                    (self.low*0.1)) - self.vwap)/(self.open - ((self.high + self.low)/2))), 11), 7))*-1)

        return data

    def alpha_088(self, n: int = 20) -> np.ndarray:
        """
        (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
        """
        data = (self.close - delay(self.close, n))/delay(self.close, n)*100
        return data

    def alpha_089(self, n: int = 10) -> np.ndarray:
        """
        2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
        """

        sma13 = ts_sma(self.close, 13, 2)
        sma27 = ts_sma(self.close, 27, 2)

        data = 2*(sma13 - sma27 - ts_sma((sma13 - sma27), n, 2))
        return data

    def alpha_090(self, n: int = 5) -> np.ndarray:
        """
        (RANK(CORR(RANK(VWAP),RANK(VOLUME),5))*-1)
        """

        data = (rank(ts_corr(rank(self.vwap), rank(self.volume), n))*-1)

        return data

    def alpha_091(self, n: int = 5) -> np.ndarray:
        """
        ((RANK((CLOSE-MAX(CLOSE,5)))*RANK(CORR((MEAN(VOLUME,40)),LOW,5)))-1)
        """

        data = ((rank((self.close - ts_max(self.close, 5))) *
                 rank(ts_corr((ts_mean(self.volume, 20)), self.low, 5)))*-1)
        return data

    def alpha_092(self, n: int = 5) -> np.ndarray:
        """
        (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1)
        """
        data = (ts_max(rank(decay_linear(delta(((self.close*0.35) + (self.vwap*0.65)), 2), 3)),
                       ts_rank(decay_linear(abs(ts_corr((ts_mean(self.volume, 30)), self.close, 13)), 5), 15))*-1)
        return data

    def alpha_093(self, n: int = 20) -> np.ndarray:
        """
        SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
        """
        data = ts_sum(np.where(self.open >= delay(self.open, 1), 0, maximum(
            (self.open - self.low), (self.open - delay(self.open, 1)))), n)
        return data

    def alpha_094(self, n: int = 30) -> np.ndarray:
        """
        SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
        """
        data = ts_sum(np.where((self.close > delay(self.close, 1)), self.volume,
                               np.where((self.close < delay(self.close, 1)), -self.volume, 0)), n)

        return data

    def alpha_095(self, n: int = 20) -> np.ndarray:
        """
        STD(AMOUNT,20)
        """
        data = ts_stddev((self.close*self.volume), n)
        return data

    def alpha_096(self, n: int = 20) -> np.ndarray:
        """
        SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
        """
        data = ts_sma(ts_sma((self.close - ts_min(self.low, 9)) /
                             (ts_max(self.high, 9)-ts_min(self.low, 9))*100, 3, 1), 3, 1)
        return data

    def alpha_097(self, n: int = 10) -> np.ndarray:
        """
        STD(VOLUME,10)
        """

        data = ts_stddev(self.volume, n)
        return data

    def alpha_098(self, n: int = 20) -> np.ndarray:
        """
        ((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))\<0.05)||((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))==0.05))?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3)))
        """
        cond1 = delta(ts_mean(self.close, n), n) / \
            delay(self.close, n) <= 0.05
        v1 = (-1*(self.close - ts_min(self.close, n)))
        v2 = (-1*delta(self.close, 3))

        data = np.where(cond1, v1, v2)
        return data

    def alpha_099(self, n: int = 5) -> np.ndarray:
        """
        (-1*RANK(COVIANCE(RANK(CLOSE),RANK(VOLUME),5)))
        """

        data = (-1*rank(ts_cov(rank(self.close), rank(self.volume), n)))
        return data

    def alpha_100(self, n: int = 20) -> np.ndarray:
        """
        STD(VOLUME,20)
        """
        data = ts_stddev(self.volume, n)
        return data

    def alpha_101(self, n: int = 15) -> np.ndarray:
        """
        ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))
        """

        data = rank(ts_corr(self.close, ts_sum(
            ts_mean(self.volume, 30), 37), n))

        return data

    def alpha_102(self, n: int = 6) -> np.ndarray:
        """
        SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
        """

        data = ts_sma(maximum(self.volume - delay(self.volume, 1), 0), n, 1) / \
            ts_sma(abs(self.volume - delay(self.volume, 1)), n, 1)*100

        return data

    def alpha_103(self, n: int = 20) -> np.ndarray:
        """
        ((20-LOWDAY(LOW,20))/20)*100
        """
        data = ((n-ts_lowday(self.low, n))/n)*100
        return data

    def alpha_104(self, n: int = 20) -> np.ndarray:
        """
        (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20))))
        """

        data = (-1*(delta(ts_corr(self.high, self.volume, 5), 5)
                    * rank(ts_stddev(self.close, n))))
        return data

    def alpha_105(self, n: int = 10) -> np.ndarray:
        """
        (-1*CORR(RANK(OPEN),RANK(VOLUME),10))
        """

        data = (-1*ts_corr(rank(self.open), rank(self.volume), n))
        return data

    def alpha_106(self, n: int = 20) -> np.ndarray:
        """
        CLOSE-DELAY(CLOSE,20)
        """
        data = self.close - delay(self.close, n)
        return data

    def alpha_107(self, n: int = 20) -> np.ndarray:
        """
        (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1))))
        """

        data = (((-1*rank((self.open - delay(self.high, 1))))*rank((self.open -
                                                                    delay(self.close, 1))))*rank((self.open - delay(self.low, 1))))
        return data

    def alpha_108(self, n: int = 20) -> np.ndarray:
        """
        ((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1)
        """
        data = ((rank((self.high - ts_min(self.high, 2))) **
                 rank(ts_corr((self.vwap), (ts_mean(self.volume, 120)), 6)))*-1)
        return data

    def alpha_109(self, n: int = 10) -> np.ndarray:
        """
        SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
        """
        data = ts_sma((self.high - self.low), n, 2) / \
            ts_sma(ts_sma((self.high-self.low), n, 2), n, 2)
        return data

    def alpha_110(self, n: int = 10) -> np.ndarray:
        """
        SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
        """
        data = ts_sum(maximum(0, self.high - delay(self.close, 1)), 20) / \
            ts_sum(maximum(0, delay(self.close, 1) - self.low), 20)*100
        return data

    def alpha_111(self, n: int = 4) -> np.ndarray:
        """
        SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
        """
        data = ts_sma(self.volume*((self.close - self.low) - (self.high - self.close))/(self.high - self.low), 11, 2) - \
            ts_sma(self.volume*((self.close - self.low) -
                                (self.high - self.close))/(self.high - self.low), 4, 2)

        return data

    def alpha_112(self, n: int = 12) -> np.ndarray:
        """
        (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
        """
        cond1 = (self.close - delay(self.close, 1) > 0)
        cond2 = (self.close - delay(self.close, 1) < 0)
        valuse1 = self.close - delay(self.close, 1)

        data = (ts_sum(np.where(cond1, valuse1, 0), n) - ts_sum(np.where(cond2, abs(valuse1), 0), n)) / \
            (ts_sum(np.where(cond1, valuse1, 0), n) +
             ts_sum(np.where(cond2, abs(valuse1), 0), n))

        return data

    def alpha_113(self, n: int = 10) -> np.ndarray:
        """
        (-1*((RANK((SUM(DELAY(CLOSE,5),20)/20))*CORR(CLOSE,VOLUME,2))*RANK(CORR(SUM(CLOSE,5),SUM(CLOSE,20),2))))
        """
        data = (-1*((rank((ts_sum(delay(self.close, 5), 20)/20))*ts_corr(self.close,
                                                                         self.volume, 2))*rank(ts_corr(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))))
        return data

    def alpha_114(self, n: int = 5) -> np.ndarray:
        """
        ((RANK(DELAY(((HIGH-LOW)/(SUM(CLOSE,5)/5)),2))*RANK(RANK(VOLUME)))/(((HIGH-LOW)/(SUM(CLOSE,5)/5))/(VWAP-CLOSE)))
        """
        data = ((rank(delay(((self.high - self.low)/(ts_sum(self.close, n)/n)), 2))*rank(rank(self.volume))
                 )/(((self.high - self.low)/(ts_sum(self.close, n)/n))/(self.vwap - self.close)))
        return data

    def alpha_115(self, n: int = 10) -> np.ndarray:
        """
        (RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)))
        """
        data = (rank(ts_corr(((self.high*0.9)+(self.close*0.1)), ts_mean(self.volume, 30), 10))
                ** rank(ts_corr(ts_rank(((self.high + self.low)/2), 4), ts_rank(self.volume, 10), 7)))
        return data

    def alpha_116(self, n: int = 20) -> np.ndarray:
        """
        REGBETA(CLOSE,SEQUENCE,20)
        """
        # TODO
        data = reg_beta(self.close, sequence(n), n)
        return data

    def alpha_117(self, n: int = 32) -> np.ndarray:
        """
        ((TSRANK(VOLUME,32)*(1-TSRANK(((CLOSE+HIGH)-LOW),16)))(1-TSRANK(RET,32)))
        """
        data = ((ts_rank(self.volume, n)*(1 - ts_rank(((self.close +
                                                        self.high) - self.low), 16)))(1-ts_rank(self.returns, n)))
        return data

    def alpha_118(self, n: int = 20) -> np.ndarray:
        """
        SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
        """

        data = ts_sum((self.high - self.open), n) / \
            ts_sum((self.open - self.low), n)*100
        return data

    def alpha_119(self, n: int = 10) -> np.ndarray:
        """
        (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))
        """
        data = (rank(decay_linear(ts_corr(self.vwap, ts_sum(ts_mean(self.volume, 5), 26), 5), 7)) -
                rank(decay_linear(ts_rank(ts_min(ts_corr(rank(self.open), rank(ts_mean(self.volume, 15)), 21), 9), 7), 8)))

        return data

    def alpha_120(self, n: int = 10) -> np.ndarray:
        """
        (RANK((VWAP-CLOSE))/RANK((VWAP+CLOSE)))
        """
        data = (rank((self.vwap - self.close))/rank((self.vwap + self.close)))
        return data

    def alpha_121(self, n: int = 10) -> np.ndarray:
        """
        ((RANK((VWAP-MIN(VWAP,12)))^TSRANK(CORR(TSRANK(VWAP,20),TSRANK(MEAN(VOLUME,60),2),18),3))*-1)
        """
        data = ((rank((self.vwap-ts_min(self.vwap, 12)))**ts_rank(ts_corr(
            ts_rank(self.vwap, 20), ts_rank(ts_mean(self.volume, 60), 2), 18), 3))*-1)
        return data

    def alpha_122(self, n: int = 10) -> np.ndarray:
        """
        (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
        """

        data = (ts_sma(ts_sma(ts_sma(log(self.close), 13, 2), 13, 2), 13, 2) - delay(ts_sma(ts_sma(ts_sma(log(self.close),
                                                                                                          13, 2), 13, 2), 13, 2), 1))/delay(ts_sma(ts_sma(ts_sma(log(self.close), 13, 2), 13, 2), 13, 2), 1)
        return data

    def alpha_123(self, n: int = 10) -> np.ndarray:
        """
        ((RANK(CORR(SUM(((HIGH + LOW) / 2),20), SUM(MEAN(VOLUME, 60), 20), 9)) < RANK(CORRLOW, VOLUME, Alpha123 6)))*-1)
        """
        data = ((rank(ts_corr(ts_sum(((self.high + self.low) / 2), 20), ts_sum(
            ts_mean(self.volume, 60), 20), 9)) < rank(ts_corr(self.low, self.volume, 6)))*-1)
        return data

    def alpha_124(self, n: int = 10) -> np.ndarray:
        """
        (CLOSE-VWAP)/DECAYLINEAR(RANK(TSMAX(CLOSE,30)),2)
        """
        data = (self.close - self.vwap) / \
            decay_linear(rank(ts_max(self.close, 30)))
        return data

    def alpha_125(self, n: int = 10) -> np.ndarray:
        """
        (RANK(DECAYLINEAR(CORR((VWAP),MEAN(VOLUME,80),17),20))/RANK(DECAYLINEAR(DELTA(((CLOSE*0.5)+(VWAP*0.5)),3),16)))
        """
        data = (rank(decay_linear(ts_corr((self.vwap), ts_mean(self.volume, 80), 17), 20)
                     )/rank(decay_linear(delta(((self.close*0.5)+(self.vwap*0.5)), 3), 16)))
        return data

    def alpha_126(self, n: int = 10) -> np.ndarray:
        """
        (CLOSE+HIGH+LOW)/3
        """
        data = (self.close + self.high + self.low)/3

        return data

    def alpha_127(self, n: int = 10) -> np.ndarray:
        """
        (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)
        """
        data = (ts_mean((100*(self.close - ts_max(self.close, 12)) /
                         (ts_max(self.close, 12)))**2))**(1/2)
        return data

    def alpha_128(self, n: int = 14) -> np.ndarray:
        """
        100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3, 1)?(HIGH+LOW+CLOSE)/3 *VOLUME:0,14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY(HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3 *VOLUME:0), 14)))
        """
        triple_price = (self.high + self.low + self.close)/3
        data = 100 - (100/(1 + ts_sum(np.where((triple_price > delay(triple_price, 1)), triple_price*self.volume, 0),
                                      n)/ts_sum(np.where(triple_price < delay(triple_price, 1), triple_price*self.volume, 0), n)))

        return data

    def alpha_129(self, n: int = 12) -> np.ndarray:
        """
        SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
        """
        data = ts_sum(np.where((self.close - delay(self.close, 1) < 0),
                               abs(self.close - delay(self.close, 1)), 0), n)

        return data

    def alpha_130(self, n: int = 10) -> np.ndarray:
        """
        (RANK(DECAYLINEAR(CORR(((HIGH+LOW)/2),MEAN(VOLUME,40),9),10))/RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),7),3)))
        """
        data = (rank(decay_linear(ts_corr(((self.high+self.low)/2), ts_mean(self.volume, 40),
                                          9), 10))/rank(decay_linear(ts_corr(rank(self.vwap), rank(self.volume), 7), 3)))

        return data

    def alpha_131(self, n: int = 10) -> np.ndarray:
        """
        (RANK(DELAT(VWAP,1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))
        """
        data = (rank(delta(self.vwap, 1)) **
                ts_rank(ts_corr(self.close, ts_mean(self.volume, 50), 18), 18))
        return data

    def alpha_132(self, n: int = 20) -> np.ndarray:
        """
        MEAN(AMOUNT,20)
        """
        data = ts_mean(self.close*self.volume, n)
        return data

    def alpha_133(self, n: int = 20) -> np.ndarray:
        """
        ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
        """
        data = ((n - ts_highday(self.high, n))/n) * \
            100-((n - ts_lowday(self.low, n))/n)*100
        return data

    def alpha_134(self, n: int = 12) -> np.ndarray:
        """
        (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
        """
        data = (self.close - delay(self.close, n)) / \
            delay(self.close, n)*self.volume
        return data

    def alpha_135(self, n: int = 20) -> np.ndarray:
        """
        SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
        """
        data = ts_sma(delay(self.close/delay(self.close, n), 1), n, 1)
        return data

    def alpha_136(self, n: int = 10) -> np.ndarray:
        """
        ((-1*RANK(DELTA(RET,3)))*CORR(OPEN,VOLUME,10))
        """
        data = ((-1*rank(delta(self.returns, 3))) *
                ts_corr(self.open, self.volume, n))
        return data

    def alpha_137(self, n: int = 10) -> np.ndarray:
        """
        16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) &ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
        """
        # AHDC = abs(self.high - delay(self.close,1))
        # ALDC = abs(self.close - delay(self.close,1))
        # ADCO = abs(delay(self.close,1) - delay(self.open,1))
        # AHDL = abs(self.high - delay(self.low,1))

        # data = 16*(self.close - delay(self.close,1) + (self.close - self.open)/2 + delay(self.close,1) - delay(self.open,1))/((AHDC > ALDC & AHDC > AHDL?AHDC+ALDC/2+ADCO/4:(ALDC>AHDL & ALDC>AHDC?ALDC+AHDC/2+ADCO/4:AHDL+ADCO/4)))*max(AHDC,ALDC)
        pass

    def alpha_138(self, n: int = 10) -> np.ndarray:
        """
        ((RANK(DECAYLINEAR(DELTA(((LOW * 0.7 + (VWAP *0.3))),3), 20)) - \
         TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5),19), 16),7))*-1)
        """
        data = ((rank(decay_linear(delta(((self.low * 0.7 + (self.vwap * 0.3))), 3), 20)) - ts_rank(decay_linear(
            ts_rank(ts_corr(ts_rank(self.low, 8), ts_rank(ts_mean(self.volume, 60), 17), 5), 19), 16), 7))*-1)
        return data

    def alpha_139(self, n: int = 10) -> np.ndarray:
        """
        (-1*CORR(OPEN,VOLUME,10))
        """
        data = (-1*ts_corr(self.open, self.volume, n))
        return data

    def alpha_140(self, n: int = 10) -> np.ndarray:
        """
        MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)- (RANK(HIGH + RANK(CLOSE))), 8)),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))
        """
        data = minimum(rank(decay_linear((rank(self.open) + rank(self.low) - (rank(self.high) + rank(self.close))), 8)),
                       ts_rank(decay_linear(ts_corr(ts_rank(self.close, 8), ts_rank(ts_mean(self.volume, 60), 20), 8), 7), 3))
        return data

    def alpha_141(self, n: int = 9) -> np.ndarray:
        """
        (RANK(CORR(RANK(HIGH),RANK(MEAN(VOLUME,15)),9))*-1)
        """
        data = (rank(ts_corr(rank(self.high), rank(ts_mean(self.volume, 15)), n))*-1)
        return data

    def alpha_142(self, n: int = 10) -> np.ndarray:
        """
        (((-1*RANK(TSRANK(CLOSE,10)))*RANK(DELTA(DELTA(CLOSE,1),1)))*RANK(TSRANK((VOLUME/MEAN(VOLUME,20)),5)))
        """
        data = (((-1*rank(ts_rank(self.close, 10)))*rank(delta(delta(self.close, 1), 1)))
                * rank(ts_rank((self.volume/ts_mean(self.volume, 20)), 5)))
        return data

    def alpha_143(self, n: int = 10) -> np.ndarray:
        """
        CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
        """
        # TODO
        pass

    def alpha_144(self, n: int = 10) -> np.ndarray:
        """
        SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)
        """
        # TODO
        pass

    def alpha_145(self, n: int = 10) -> np.ndarray:
        """
        (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
        """
        data = (ts_mean(self.volume, 9) - ts_mean(self.volume, 26)) / \
            ts_mean(self.volume, 12)*100
        return data

    def alpha_146(self, n: int = 10) -> np.ndarray:
        """
        MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOS E-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,60);
        """
        v1 = (self.close - delay(self.close, 1))/delay(self.close, 1)
        data = ts_mean(v1-ts_sma(v1,61,2),20)*(v1-ts_sma(v1,61,2))/ts_sma((v1-(v1-ts_sma(v1,61,2)))**2,60,2)

        return data

    def alpha_147(self, n: int = 12) -> np.ndarray:
        """
        REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
        """
        data = reg_beta(ts_mean(self.close, n), sequence(n))
        return data

    def alpha_148(self, n: int = 10) -> np.ndarray:
        """
        ((RANK(CORR((OPEN),SUM(MEAN(VOLUME,60),9),6))<RANK((OPEN-TSMIN(OPEN,14))))*-1)
        """
        data = ((rank(ts_corr((self.open), ts_sum(ts_mean(self.volume, 60), 9), 6)) < rank(
            (self.open - ts_min(self.open, 14))))*-1)
        return data

    def alpha_149(self, n: int = 10) -> np.ndarray:
        """
        REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)
        """
        # TODO
        pass

    def alpha_150(self, n: int = 10) -> np.ndarray:
        """
        (CLOSE+HIGH+LOW)/3*VOLUME
        """
        data = (self.close + self.high + self.low)/3*self.volume
        return data

    def alpha_151(self, n: int = 20) -> np.ndarray:
        """
        SMA(CLOSE-DELAY(CLOSE,20),20,1)
        """
        data = ts_sma(self.close - delay(self.close, n), n, 1)
        return data

    def alpha_152(self, n: int = 10) -> np.ndarray:
        """
        SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12) - \
            MEAN(DELAY(SMA(DELAY(CLOSE/DELAY (CLOSE,9),1),9,1),1),26),9,1)
        """
        data = ts_sma(ts_mean(delay(ts_sma(delay(self.close/delay(self.close, 9), 1), 9, 1), 1), 12) -
                      ts_mean(delay(ts_sma(delay(self.close/delay(self.close, 9), 1), 9, 1), 1), 26), 9, 1)
        return data

    def alpha_153(self, n: int = 3) -> np.ndarray:
        """
        (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
        """
        data = (ts_mean(self.close, n) + ts_mean(self.close, 2*n) +
                ts_mean(self.close, 4*n) + ts_mean(self.close, 8*n))/4
        return data

    def alpha_154(self, n: int = 10) -> np.ndarray:
        """
        (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18)))
        """
        data = (((self.vwap - ts_min(self.vwap, n))) <
                (ts_corr(self.vwap, ts_mean(self.volume, n), n)))
        return data

    def alpha_155(self, n: int = 10) -> np.ndarray:
        """
        SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
        """
        data = ts_sma(self.volume, 13, 2) - ts_sma(self.volume, 27, 2) - ts_sma(ts_sma(self.volume, 13, 2) -
                                                                                ts_sma(self.volume, 27, 2), n, 2)
        return data

    def alpha_156(self, n: int = 10) -> np.ndarray:
        """
        (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))-1),3)))*-1)
        """
        data = (maximum(rank(decay_linear(delta(self.vwap, 5), 3)), rank(decay_linear(
            ((delta(((self.open*0.15)+(self.low*0.85)), 2)/((self.open*0.15)+(self.low*0.85)))-1), 3)))*-1)

        return data

    def alpha_157(self, n: int = 10) -> np.ndarray:
        """
        (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),
         5) +TSRANK(DELAY((-1*RET),6),5))
        """
        data = (ts_min(ts_prod(rank(rank(log(ts_sum(ts_min(rank(-1*rank(delta(self.close, 1))), 5), 2)))),
                               1), 5) + ts_rank(delay((-1*self.returns), 6), 5))
        return data

    def alpha_158(self, n: int = 15) -> np.ndarray:
        """
        ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
        """
        data = ((self.high - ts_sma(self.close, n, 2)) -
                (self.low - ts_sma(self.close, n, 2)))/self.close
        return data

    def alpha_159(self, n: int = 10) -> np.ndarray:
        """
        ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CL OSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,D ELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
        """

        MN = maximum(self.high, delay(self.close, 1)) - \
            minimum(self.low, delay(self.close, 1))
        MLDC = minimum(self.low, delay(self.close, 1))

        data = ((self.close - ts_sum(MLDC, 6))/ts_sum(MN, 6)*12*24+(self.close - ts_sum(MLDC, 12)) /
                ts_sum(MN, 12)*6*24+(self.close - ts_sum(MLDC, 24))/ts_sum(MN, 24)*6*24)*100/(6*12+6*24+12*24)
        return data

    def alpha_160(self, n: int = 20) -> np.ndarray:
        """
        SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
        """
        data = ts_sma(np.where(self.close <= delay(
            self.close, 1), ts_stddev(self.close, n), 0), n, 1)
        return data

    def alpha_161(self, n: int = 12) -> np.ndarray:
        """
        MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
        """
        data = ts_mean(maximum(maximum((self.high - self.low), abs(delay(self.close,
                                                                         1) - self.high)), abs(delay(self.close, 1) - self.low)), n)
        return data

    def alpha_162(self, n: int = 12) -> np.ndarray:
        """
        (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/ \
         (MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)- \
          MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12, 1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
        """
        v1 = ts_sma(maximum(self.close - delay(self.close, 1), 0), n, 1)
        v2 = ts_sma(abs(self.close - delay(self.close, 1)), n, 1)

        data = (v1/v2*100-ts_min(v1/v2*100, n)) / \
            (ts_max(v1/v2*100, n) - ts_min(v1/v2*100, n))
        return data

    def alpha_163(self, n: int = 20) -> np.ndarray:
        """
        RANK(((((-1*RET)*MEAN(VOLUME,20))*VWAP)*(HIGH-CLOSE)))
        """
        data = rank(((((-1*self.returns)*ts_mean(self.volume, n))
                    * self.vwap)*(self.high - self.close)))
        return data

    def alpha_164(self, n: int = 10) -> np.ndarray:
        """
        SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-D ELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)
        """
        data = ts_sma((np.where((self.close > delay(self.close, 1)), 1/(self.close - delay(self.close, 1)), 1) - ts_min(np.where(
            (self.close > delay(self.close, 1)), 1/(self.close - delay(self.close, 1)), 1), 12))/(self.high - self.low)*100, 13, 2)
        return data

    def alpha_165(self, n: int = 48) -> np.ndarray:
        """
        MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
        """
        ac = self.close - ts_mean(self.close, n)
        data = ts_max(ac, n) - ts_min(ac, n)/ts_stddev(self.close, n)
        return data

    # def alpha_166(self, n: int = 10) -> np.ndarray:
    #     """
    #     -20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
    #     """
    #     data = -20*(20-1)**1.5*ts_sum(self.close/delay(self.close, 1) - 1 - ts_mean(self.close/delay(
    #         self.close, 1)-1, 20), 20)/((20-1)*(20-2)*(ts_sum((self.close/delay(self.close, 1), 20)**2, 20))**1.5)
    #     return data

    def alpha_167(self, n: int = 12) -> np.ndarray:
        """
        SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12
        """
        data = ts_sum(np.where((self.close - delay(self.close, 1) > 0),
                               (self.close - delay(self.close, 1)), 0), n)
        return data

    def alpha_168(self, n: int = 20) -> np.ndarray:
        """
        (-1*VOLUME/MEAN(VOLUME,20))
        """
        data = (-1*self.volume/ts_mean(self.volume, n))
        return data

    def alpha_169(self, n: int = 10) -> np.ndarray:
        """
        SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12) - \
            MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1), 26),10,1)
        """
        data = ts_sma(ts_mean(delay(ts_sma(self.close - delay(self.close, 1), 9, 1), 1), 12) -
                      ts_mean(delay(ts_sma(self.close - delay(self.close, 1), 9, 1), 1), 26), 10, 1)
        return data

    def alpha_170(self, n: int = 10) -> np.ndarray:
        """
        ((((RANK((1/CLOSE))VOLUME)/MEAN(VOLUME,20))((HIGH*RANK((HIGH-CLOSE)))/(SUM(HIGH,5)/5)))-RANK((VWAP-DELAY(VWAP,5))))
        """
        data = ((((rank((1/self.close))*self.volume)/ts_mean(self.volume, 20))*((self.high *
                                                                                 rank((self.high - self.close)))/(ts_sum(self.high, 5)/5))) - rank((self.vwap - delay(self.vwap, 5))))
        return data

    def alpha_171(self, n: int = 10) -> np.ndarray:
        """
        ((-1*((LOW-CLOSE)(OPEN^5)))/((CLOSE-HIGH)(CLOSE^5)))
        """
        data = ((-1*((self.low - self.close)*(self.open**5))) /
                ((self.close - self.high)*(self.close**5)))

        return data

    def alpha_172(self, n: int = 10) -> np.ndarray:
        """
        MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
        """
        data = ts_mean(abs(ts_sum(np.where(((self.LD > 0) & (self.LD > self.HD)), self.LD, 0), 14))*100/ts_sum(self.TR, 14) - ts_sum(np.where(((self.HD > 0) & (self.HD >
                       self.LD)), self.HD, 0), 14)*100/(ts_sum(self.TR, 14) + ts_sum(np.where(((self.HD > 0) & (self.HD > self.LD)), self.HD, 0), 14)/ts_sum(self.TR, 14))*100, 6)
        return data

    def alpha_173(self, n: int = 13) -> np.ndarray:
        """
        3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2);
        """
        data = 3*ts_sma(self.close, n, 2) - 2*ts_sma(ts_sma(self.close, n, 2),
                                                     n, 2) + ts_sma(ts_sma(ts_sma(log(self.close), n, 2), n, 2), n, 2)
        return data

    def alpha_174(self, n: int = 20) -> np.ndarray:
        """
        SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
        """
        data = ts_sma(np.where((self.close > delay(self.close, 1)),
                               ts_stddev(self.close, n), 0), n, 1)
        return data

    def alpha_175(self, n: int = 6) -> np.ndarray:
        """
        MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
        """
        data = ts_mean(maximum(maximum((self.high - self.low), abs(delay(self.close,
                                                                         1) - self.high)), abs(delay(self.close, 1) - self.low)), n)
        return data

    def alpha_176(self, n: int = 12) -> np.ndarray:
        """
        CORR(RANK(((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12)))),RANK(VOLUME),6)
        """
        data = ts_corr(rank(((self.close - ts_min(self.low, 12)) /
                             (ts_max(self.high, 12) - ts_min(self.low, 12)))), rank(self.volume), n)
        return data

    def alpha_177(self, n: int = 20) -> np.ndarray:
        """
        ((20-HIGHDAY(HIGH,20))/20)*100
        """
        data = ((n - ts_highday(self.high, n))/n)*100
        return data

    def alpha_178(self, n: int = 10) -> np.ndarray:
        """
        (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
        """
        data = (self.close-delay(self.close, 1)) / \
            delay(self.close, 1)*self.volume
        return data

    def alpha_179(self, n: int = 12) -> np.ndarray:
        """
        (RANK(CORR(VWAP,VOLUME,4))*RANK(CORR(RANK(LOW),RANK(MEAN(VOLUME,50)),12)))
        """
        data = (rank(ts_corr(self.vwap, self.volume, 4)) *
                rank(ts_corr(rank(self.low), rank(ts_mean(self.volume, 50)), n)))

        return data

    def alpha_180(self, n: int = 10) -> np.ndarray:
        """
        ((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME)))
        """
        data = np.where((ts_mean(self.volume, 20) < self.volume), (-1*ts_rank(
            abs(delta(self.close, 7)), 60))*sign(delta(self.close, 7)), -1*self.volume)
        return data

    # def alpha_181(self, n: int = 10) -> np.ndarray:
    #     """
    #     SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)
    #     """
    #     # TODO
    #     pass

    # def alpha_182(self, n: int = 10) -> np.ndarray:
    #     """
    #     COUNT((CLOSE>OPEN&BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN&BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20
    #     """
    #     # TODO
    #     pass

    def alpha_183(self, n: int = 24) -> np.ndarray:
        """
        MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
        """
        ac = ts_sumac((self.close - ts_mean(self.close, n)), n)
        data = (ts_max(ac, n) - ts_min(ac, n))/ts_stddev(self.close, n)
        return data

    def alpha_184(self, n: int = 20) -> np.ndarray:
        """
        (RANK(CORR(DELAY((OPEN-CLOSE),1),CLOSE,200))+RANK((OPEN-CLOSE)))
        """
        data = (rank(ts_corr(delay((self.open - self.close), 1),
                             self.close, n)) + rank((self.open - self.close)))
        return data

    def alpha_185(self, n: int = 10) -> np.ndarray:
        """
        RANK((-1*((1-(OPEN/CLOSE))^2)))
        """
        data = rank((-1*((1-(self.open/self.close))**2)))
        return data

    def alpha_186(self, n: int = 10) -> np.ndarray:
        """
        (MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
        """
        v1 = ts_sum(
            np.where(((self.LD > 0) & (self.LD > self.HD)), self.LD, 0), 14)
        v2 = ts_sum(
            np.where(((self.HD > 0) & (self.HD > self.LD)), self.HD, 0), 14)
        SUMTR = ts_sum(self.TR, 14)

        data = (ts_mean(abs(v1*100/SUMTR - v2*100/SUMTR)/(v1*100/SUMTR+v2*100/SUMTR)*100, 6) +
                delay(ts_mean(abs(v1*100/SUMTR-v2*100/SUMTR)/(v1*100/SUMTR+v2*100/SUMTR)*100, 6), 6))/2
        return data

    def alpha_187(self, n: int = 20) -> np.ndarray:
        """
        SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
        """
        data = ts_sum(np.where((self.open <= delay(self.open, 1)), 0, maximum(
            (self.high - self.open), (self.open - delay(self.open, 1)))), n)

        return data

    def alpha_188(self, n: int = 11) -> np.ndarray:
        """
        ((HIGH-LOW-SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
        """
        data = ((self.high - self.low - ts_sma(self.high - self.low, n, 2)) /
                ts_sma(self.high - self.low, n, 2))*100
        return data

    def alpha_189(self, n: int = 6) -> np.ndarray:
        """
        MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
        """
        data = ts_mean(abs(self.close - ts_mean(self.close, n)), n)
        return data

    def alpha_190(self, n: int = 10) -> np.ndarray:
        """
        LOG((COUNT(CLOSE/DELAY(CLOSE,1)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)(SUMIF(((CLOSE/DELAY(CLOSE,1)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE,1)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))(SUMIF((CLOSE/DELAY(CLOSE,1)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
        """
        pass

    def alpha_191(self, n: int = 5) -> np.ndarray:
        """
        ((CORR(MEAN(VOLUME,20),LOW,5)+((HIGH+LOW)/2))-CLOSE)
        """
        data = ((ts_corr(ts_mean(self.volume, 20), self.low, n) +
                 ((self.high + self.low)/2)) - self.close)

        return data


# -------------------------------------
# 高頻因子
# -------------------------------------
class HFTFactors:

    def __init__(self):
        pass

    @ staticmethod
    @ njit
    def rwr(last_price, n: int = 120) -> np.ndarray:
        """收益波动率因子"""
        rwr = np.empty(len(last_price))
        rwr[:] = np.nan  # Initialize with NaNs

        for i in range(n, len(last_price)):

            rwr[i] = (last_price[i] - last_price[i-n]) / \
                (np.max(last_price[i-n:i]) - np.min(last_price[i-n:i]))

        return rwr

    @ staticmethod
    @ njit
    def z_t(last_price, ask_price, bid_price) -> np.ndarray:
        """
        计算 Z_t 因子的函数

        参数:
        - last_price: 最新成交价
        - ask_price: 卖一价
        - bid_price: 买一价
        返回:
        - Z_t 因子值
        """
        return np.log(last_price) - np.log((ask_price + bid_price)/2)

    @ staticmethod
    @ njit
    def slpoe(bid_price, ask_price, bid_qty, ask_qty) -> np.ndarray:
        """
        计算 Slope 因子的函数

        参数:
        - bid_price: 买一价
        - ask_price: 卖一价
        - bid_qty: 买一量
        - ask_qty: 卖一量

        返回:
        - Slope 因子值
        """

        return (ask_price - bid_price)/(bid_qty + ask_qty)*2

    @ staticmethod
    @ njit
    def voi(bid_price, ask_price, bid_qty, ask_qty, volume) -> np.ndarray:
        """
        计算 voi 訂單失衡

        参数:
        - bid_price: 买一价
        - ask_price: 卖一价
        - bid_qty: 买一量
        - ask_qty: 卖一量
        - volume: 成交量
        返回:
        - voi 因子值
        """

        bid_sub_price = bid_price[1:] - bid_price[:-1]
        ask_sub_price = ask_price[1:] - ask_price[:-1]
        bid_sub_volume = bid_qty[1:] - bid_qty[:-1]
        ask_sub_volume = ask_qty[1:] - ask_qty[:-1]

        # 使用 np.where 更新 bid_volume_change 和 ask_volume_change
        bid_volume_change = np.where(bid_sub_price < 0, 0, np.where(
            bid_sub_price > 0, bid_qty[1:], bid_sub_volume))

        ask_volume_change = np.where(ask_sub_price > 0, 0, np.where(
            ask_sub_price < 0, ask_qty[1:], ask_sub_volume))

        voi = np.where((volume[1:] == 0), 0,
                       (bid_volume_change - ask_volume_change) / volume[1:])

        voi[np.isposinf(voi)] = np.nan
        voi[np.isnan(voi)] = 0

        # Create a new array with nan at the beginning
        new_voi = np.empty(len(voi) + 1)
        new_voi[0] = np.nan
        new_voi[1:] = voi

        return new_voi

    @ staticmethod
    @ njit
    def mbp(turn_over, volume, bid_price, ask_price) -> np.ndarray:
        """市价偏离度 Mid-Price Basis"""

        tp = turn_over/volume
        tp[np.isinf(tp)] = np.nan

        nan_mask = np.isnan(tp)
        last_valid = np.nan

        # 遍历数组并填充 NaN 值
        for i in range(len(tp)):
            if nan_mask[i]:
                tp[i] = last_valid
            else:
                last_valid = tp[i]

        mid = (bid_price + ask_price)/2

        mid_shifted = np.empty_like(mid)
        mid_shifted[0] = mid[0]
        mid_shifted[1:] = mid[:-1]

        return tp - (mid + mid_shifted) / 2

    @ staticmethod
    @ njit
    def positive_ratio(turn_over, last_price, ask_price, tick_nums: int = 120) -> np.ndarray:
        """"""
        buy_positive = np.empty_like(turn_over)
        buy_positive[:] = np.nan  # 将所有元素初始化为 NaN

        positive_turnover = np.zeros_like(turn_over)
        positive_turnover[1:] = np.where(
            (last_price[1:] >= ask_price[:-1]), turn_over[1:], 0)

        for i in range(tick_nums - 1, len(turn_over)):

            PTSUM = np.sum(positive_turnover[i - tick_nums + 1:i + 1])
            TOSUM = np.sum(turn_over[i - tick_nums + 1:i + 1])

            buy_positive[i] = np.where(TOSUM == 0, 0, PTSUM/TOSUM)

        return buy_positive

    @ staticmethod
    @ njit
    def negtive_ratio(turn_over, last_price, bid_price, tick_nums: int = 120) -> np.ndarray:
        """"""
        sell_positive = np.empty_like(turn_over)
        sell_positive[:] = np.nan  # 将所有元素初始化为 NaN

        negtive_turnover = np.zeros_like(turn_over)
        negtive_turnover[1:] = np.where(
            (last_price[1:] <= bid_price[:-1]), turn_over[1:], 0)

        for i in range(tick_nums - 1, len(turn_over)):

            NTSUM = np.sum(negtive_turnover[i - tick_nums + 1:i + 1])
            TOSUM = np.sum(turn_over[i - tick_nums+1: i + 1])

            sell_positive[i] = np.where(TOSUM == 0, 0, NTSUM/TOSUM)

        return sell_positive

    @ staticmethod
    @ njit
    def speculation(volume, open_interest, n: int = 120) -> np.ndarray:
        """Speculation"""

        return ts_sum(volume, n)/open_interest

    @ staticmethod
    @ njit
    def diff_interest_sum(diff_interest, n: int = 120) -> np.ndarray:
        """"""
        return ts_sum(diff_interest, n)

    @ staticmethod
    @ njit
    def avg_price(total_turnover, total_volume, last_prcie) -> np.ndarray:
        """分时均价线"""

        return last_prcie/(total_turnover/total_volume)
