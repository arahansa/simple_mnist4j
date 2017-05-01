package com.arahansa.ml;

import mikera.matrixx.Matrix;
import org.springframework.stereotype.Component;

/**
 * Created by jarvis on 2017. 4. 27..
 * 교차 엔트로피 오차 구함
 */
@Component
public class J01_CostFunction {

    /**
     * 교차 엔트로피 오차
     * -np.sum(t * np.log(y+delta))
     * @param y
     * @param t
     * @return
     */
    public double getCrossEntropyErr(Matrix y, Matrix t){
        double delta = 1e-7;
        y.add(delta);
        y.log();

        t.multiply(y);
        return -t.elementSum();
    }

    /**
     * 배치용 교차 엔트로피 오차 구하기
     * 파이썬 예제와는 다르게
     * 1, X 값으로 오는 것으로 기대하자.. 소프트맥스로처리 된 값을 받는다
     * 일단 안 쓸 듯하다
     * @param y
     * @param t
     * @return
     */
    public double getCrossEntropyErr4Batch(Matrix y, Matrix t){
        // 1차원일 때의 처리는 생략. 왜냐면 매트릭스만 받는다..

        // 원핫 인코딩 처리

        return getCrossEntropyErr(y, t);
    }




}
