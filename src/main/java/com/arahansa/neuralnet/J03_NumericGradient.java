package com.arahansa.neuralnet;

import mikera.matrixx.Matrix;
import org.springframework.stereotype.Component;

import java.util.function.Function;

/**
 * Created by jarvis on 2017. 4. 27..
 */
public class J03_NumericGradient {

    /**
     * 기울기
     * @param f
     * @param m
     * @return
     */
    public Matrix getNumericGradient(Function<Matrix, Double> f, Matrix m){
        double h = 1e-4;
        Matrix grad = new Matrix(m.getShape(0), m.getShape(1)); // 같은모양복제
        for (int y = 0; y < m.getShape(1); y++) {
            for (int x = 0; x < m.getShape(0); x++) {
                double tmp_val = m.get(x,y);
                // f(x+h) 계산
                m.set(x,y, (tmp_val+h) );
                double fxh1 = f.apply(m);
                // f(x-h) 계산
                m.set(x,y, (tmp_val-h) );
                double fxh2 = f.apply(m);

                grad.set(x,y, (fxh1 - fxh2) / (2 * h) );
                // 값 복원
                m.set(x,y, tmp_val);
            }
        }
        return grad;
    }
}
