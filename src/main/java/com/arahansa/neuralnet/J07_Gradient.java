package com.arahansa.neuralnet;

import com.arahansa.neuralnet.J06_TwoLayerNet.Context;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;

import java.util.function.BiFunction;

/**
 * Created by arahansa on 2017-05-03.
 */
@Slf4j
public class J07_Gradient {
    /**
     * (w = 초기 가중치)를 받고 다음 가중치를 보여줌
     * @param w
     * @return
     */
    public static Matrix numerical_gradient(BiFunction<Matrix, Matrix, Double> f, Matrix x, Matrix t, Matrix w){
        double h = 1e-4;
        Matrix grad = new Matrix(w.getShape(0), w.getShape(1));

        // i 가 x 좌표, j 가 y 좌표
        for (int i = 0; i < w.getShape(0); i++) {
            double mibunSum = 0;
            for (int j = 0; j < w.getShape(1); j++) {
                double tmp_val = w.get(i,j);
                Matrix sampleW, sampleW1;
                // log.info("현재 ({}, {}) 번째 값은 : {}", i, j, tmp_val);

                // f(x+h) 계산
                w.set(i,j, (tmp_val+h) );
                sampleW = w.copy();
                // log.info(" 1 w : {}", w);
                double fxh1 = f.apply(x,t);

                // f(x-h) 계산
                w.set(i,j, (tmp_val-h) );
                sampleW1 = w.copy();
                // log.info(" 2 w : {}", w);
                double fxh2 = f.apply(x, t);
                final double v = (fxh1 - fxh2) / (2 * h);
                grad.set(i,j, v);
                if(Context.local.get() !=null){
                    log.info(Context.local.get()+ "{ ({}, {}) 에서의 (fxh1 - fxh2) : {}", i,j, (fxh1 - fxh2));
                    if(fxh1-fxh2==0){
                        log.info("sample w1 : {}", sampleW);
                        log.info("sample w2 : {}", sampleW1);
                    }
                    mibunSum += v;
                }
                // 값 복원
                w.set(i,j, tmp_val);
            }

            if(Context.local.get() !=null && mibunSum!=0){
                log.info(Context.local.get() + "의 {} 줄에서의 미분 총 값 : {}",i, mibunSum);
            }
        }
        return grad;
    }
}
