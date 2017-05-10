package com.arahansa.neuralnet;

import com.arahansa.data.Grad;
import com.arahansa.image.LoadMnist;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import mikera.vectorz.impl.ArraySubVector;
import org.junit.Test;
import org.springframework.util.StopWatch;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;


/**
 * Created by arahansa on 2017-05-03.
 */
@Slf4j
public class J06_TwoLayerNetTest {

    /**
     * @throws Exception
     */
    @Test
    public void 미니배치_학습구현() throws Exception{
        StopWatch watch = new StopWatch();
        watch.start();
        LoadMnist loadMnist = new LoadMnist();
        Map<String, Matrix> mnistMap = loadMnist.loadMnist();

        Matrix x_train = mnistMap.get("x_train");
        Matrix t_train = mnistMap.get("t_train");
        Matrix x_test = mnistMap.get("x_test");
        Matrix t_test = mnistMap.get("t_test");

        int iters_num = 300;
        int batch_size = 100;
        int train_size = x_train.getShape(0);
        double learning_rate  = 0.4f;
        int iter_per_epoch = train_size / batch_size;

        List<Double> train_loss_list = new ArrayList<>(iters_num);
        List<Double> train_acc_list = new ArrayList<>(iter_per_epoch);
        List<Double> test_acc_list = new ArrayList<>(iter_per_epoch);

        J06_TwoLayerNet network = new J06_TwoLayerNet(15, 20, 10, 1.0f);

        for(int i=0;i<iters_num;i++){
            Map<String, Matrix> bm = J00_Helper.getBatchMatrix(x_train, t_train, batch_size);
            Matrix x_batch = bm.get("x_batch");
            Matrix t_batch = bm.get("t_batch");

            Grad grad = network.numerical_gradient(x_batch, t_batch);
            network.renewParams(grad, learning_rate);
            Double loss = network.lossFunc.apply(x_batch, t_batch);
            train_loss_list.add(loss);

            if( i % 10 == 0){
                log.info("count : {}", i);
                double train_acc = network.accuracy(x_train, t_train);
                double test_acc = network.accuracy(x_test, t_test);

                train_acc_list.add(train_acc);
                test_acc_list.add(test_acc);
                log.info("i : {} , train acc : {}, test acc : {}", i, train_acc, test_acc);
            }
        }

        System.out.println(watch.prettyPrint());
    }


    @Test
    public void 열줄이미지가지고_백번만돌리기() throws Exception{
        double learning_rate = 0.4;
        int input_size = 20;
        J06_TwoLayerNet network = new J06_TwoLayerNet(15, input_size,
                10, new Float(learning_rate));

        List<Double> accList = new ArrayList<>();

        LoadMnist loadMnist = new LoadMnist();
        Map<Integer, Matrix> xtrainSampleMap = loadMnist.getXtrainSampleMap();
        Map<Integer, Matrix> ttrainSampleMap = loadMnist.getTtrainSampleMap();

        // 데이터 세팅
        int size = 10;
        Matrix x_batch = Matrix.create(size, 15);
        Matrix t_batch = Matrix.create(size, 10);

        for(int i=0;i<size;i++) {
            ArraySubVector xRow = xtrainSampleMap.get(i % 10).getRow(0);
            ArraySubVector tRow = ttrainSampleMap.get(i % 10).getRow(0);
            x_batch.setRow(i, xRow);
            t_batch.setRow(i, tRow);
        }

        for(int i=0;i<100;i++){
            Grad grad = network.numerical_gradient(x_batch, t_batch);
            network.renewParams(grad, learning_rate);

            if(i%10==0){
                final double accuracy = network.accuracy(x_batch, t_batch);
                log.info("acc: {}", accuracy);
                accList.add(accuracy);
            }
        }
        log.info("acc list : {}", accList);


       /* Matrix w1 = Matrix.create(15, input_size);
        w1.setElements(
                -0.5832131285233637,0.5546232317776113,-1.02258247420341,0.8243164440349637,-0.16639924876385725,-0.23332879890749791,-0.6978080401320819,-1.201138393353154,0.6678297489224253,-0.12098475252919609,-0.3636192820944814,-0.007116726710625545,0.5851253358592502,-0.06486206734020675,0.15462802416520918,1.2560328814710902,1.6170342206619672,-0.7247052795621695,-1.6738737773367534,0.11485358844937661,
                -2.1960044074720426,0.10916767641474101,-0.8575282080336422,0.18146622860526446,1.0230975164803955,2.230546314861665,0.6839562589307516,1.2765940659157402,0.6400894137026407,-0.5025554862735423,-0.7830607361970826,0.1399844182355591,0.5170155683282845,-0.4582405779393589,-0.6861103238026324,0.5384687990745262,-0.19657543178713882,0.8692159599602527,-0.06879590497139987,2.0016463211672337,
                -2.0390003112225257,-0.13501383380737403,0.092551957028291,-1.4301706064265236,-1.51646142626731,0.6527863328651854,-0.4375769095516733,-0.2616571060094534,-1.0181558623756053,0.8993809768911969,-0.0755142168431911,2.7958139075404085,-0.9579814507707247,1.5728399659861492,0.3606992172085906,0.5611924868436017,-0.554841220868448,-0.9154580477737951,1.15334814851474,0.5546185212756523,
                0.06355656334207033,-0.1564326902667807,-1.1779297458376583,0.5123709097156922,1.2344112683144597,1.486150566841667,-0.2656485208299998,-0.6582627462826944,1.2414377206727925,3.295685602219065,-0.34442274821358493,0.26789653970573846,1.0101781657401556,-0.4444460974875524,0.41878758441990216,-0.42114343670862553,-0.5039326112856248,-1.1175033451463916,-1.3255982346040183,0.8664582941259302,
                -0.9981697979694187,-0.6117009119704756,-0.8404951759143134,-0.05026642682866235,-1.392078183064078,0.38148123098229203,0.8731303871142118,-0.7971914854777031,0.037434888004452334,1.2263118234320207,-0.08686556595731332,0.07037005026753147,0.0021901832697430327,0.4363685258130466,0.7932908094622271,-0.9709463489578078,-0.5265425260318936,-0.3107699026718654,-0.12722559128549707,0.38789011619014235,
                1.7057266489173435,-0.10498255158641262,-0.7024631154258564,-0.5574046064220263,-0.4297167073684931,0.5447157609910309,0.6692689663209391,0.09165290254326851,-0.7149292707115317,0.2603879368800421,-0.14190335154359446,0.9444251008988646,-0.16289310532216156,0.7936831446227277,1.9076124162161536,1.2560071131967034,0.4744283664898206,-0.6373079311249463,-0.8862674144658899,1.1488219548262018,
                -0.7071055767788609,-1.1722219757010766,1.0641065357233566,-1.0856967496886074,0.5429434057351684,1.046131644746873,-0.48173132441914723,0.3005620147758014,0.7057986316540363,-0.339910699358982,3.0566982225170105,-1.1009172379738819,-0.35876979780913637,0.6390452190446086,0.5904786177572018,0.7967508041427749,0.30757203102045083,-0.7033609011941422,-1.3138617559610963,-0.5593599700369796,
                0.4210924757104555,0.5055570506074062,-0.6109924875934407,-1.3775663862041319,0.007036575484356437,-0.11078080511459974,0.8874614574689242,-1.2166706854410587,-0.5967172456437143,0.3641945803867415,1.1460458902828468,0.7316743555095526,1.3150046223118066,-0.5917006384234611,0.8816146520994863,0.5422387431049939,-0.004976675311718363,0.8549470780793487,-1.321977948150516,1.136582567945368,
                -0.9426481910237337,-0.8895824838760034,-0.6114599642573931,-0.9623695478063362,-0.23925517989959552,-0.5763197196534189,0.8883582229249963,1.8694276283026605,-0.2495829032138117,0.29850782075128823,-0.425353928907089,-0.19061913969037517,-1.128683213786236,2.048653254224502,1.2826827077902903,0.42633173071407693,-0.7354438233972094,0.1988099290524459,-0.428854584751412,0.018013708467837026,
                0.4099147123921482,-0.700213276624436,-1.044011330283384,-1.0049743880863193,-0.437571102390899,-0.8989489351179052,-0.3056081352582339,-0.5690269892346571,0.3853013121274834,-0.12904594802647196,0.32339448957323447,-0.40319001539888866,0.5501092636486908,-0.9265274156474073,-0.9802672252798936,0.41473532285996273,0.07826652971402591,-0.454868297919447,1.8024070792797193,-0.6573692015735509,
                1.936854734807018,-0.3781629013357878,0.6095258501379681,-2.027535207909814,0.4922727131072454,0.24549150596211372,0.9199296503290153,0.8337851991941454,0.527289442211607,0.6146670617060478,0.23690266241594118,0.5282036255654071,0.13292709477553422,-0.5319134793419927,-0.20113752691845013,-1.2866615494236788,-2.6220125848642977,-2.7468100518395353,0.9766853981755453,-0.009892207883805943,
                -0.1511812396455604,-0.19288905897904038,1.278265816424119,-0.4452538024937409,-1.0492667077407856,-0.8082958257085043,0.6049255035175471,-0.08407689873698312,0.39263718868780767,-0.36498941516890554,-0.2779422154431738,-1.027564700877942,-0.5832604198997365,0.3337681703762059,1.0103495311878004,-0.696813935626284,-1.2747832086814785,0.6334362082219487,-0.5725935547076859,0.8230490599827827,
                -0.8503978311833648,-1.3352599010383273,-0.555867402694812,0.08217368476905423,0.9037175027528601,-1.025760423996414,0.28883918599750336,0.9778129635549985,-1.4010630220738218,-1.6862379663614682,-1.0955029246231962,-0.5986080046647649,0.579509389476287,-0.4782125300852835,1.4138808503170357,-1.3765557119742227,-0.49869184368607733,1.3403371413421925,0.8078583545426257,-1.838747552215498,
                1.317786080681839,0.4726664725466116,0.7497570746023539,0.851423209935642,-0.9428367274638262,0.7330765054896953,1.2508733403013559,-0.3415506461046698,2.0700040005184217,-0.7742063231757103,-0.5718431009661537,-0.42450578264210065,-0.6129012231287408,-0.21521770545157365,-0.9656202368780213,-0.7273831074223376,1.5047988698212382,0.47874828726263546,-0.11939200766086822,-0.7977484889564701,
                -0.07189203708422678,1.5417523712456118,-0.20639314103985382,-0.9816166452965714,-0.9389386813164481,-0.3903318680309627,1.7295111888274988,0.5492312364915635,0.47168936537071837,1.5752578615836186,0.8341794797001169,1.345889874659422,0.8961770839556652,-0.6187424288977845,1.3496788291423907,0.048560565473247226,9.391440357197202E-4,-1.064305461547677,-0.530549677708072,0.6254009088920068
        );
        network.setW1(w1);

        Matrix w2 = Matrix.create(input_size, 10);
        w2.setElements(0.12832165934351278,1.7307457418865164,-0.7955517175271501,0.19705962940561667,0.5791455628924007,1.2317936856687322,-0.3984270633160795,-1.3066620482926152,-0.5979459246817268,-0.28367295788347885,
                -0.8351354988031502,-0.4234769253133356,0.2081429695448249,0.8124141199983028,-0.21545678533745774,-1.4239417004968398,1.6904424014515442,-0.6605048267651356,-0.3360248271554711,1.855273147505157,
                0.23572016333681994,0.967018393501678,1.9569337442087207,-0.22848795872248798,0.19417730373070852,0.43148779534515014,-1.9290933736540719,-0.33616051270797803,1.1472968153002252,2.567949635577401,
                -0.545833208870957,0.24572747991019186,0.10965603027782918,-0.2227829520110617,-0.07873090964425418,1.7894390905399669,0.7563932147744111,-0.8261058196366828,1.356788314304989,-0.3409166574128076,
                -2.5085485510095906,0.3219407097624265,1.3232893653415583,-1.2238935566837525,-0.8328049685720232,0.3647287105961014,1.271768612671993,0.6375754258988534,-0.9483250619861961,0.7362274782559146,
                -1.3109157181995927,0.615167745143953,-0.6783338683551938,1.1354478501378933,-0.28165165307409507,-0.06741934800698555,0.0340076433696741,0.29271571322516826,-1.5816217846030547,-0.14010275094998617,
                -1.1447375659441485,0.7481451330919384,0.08503131968410195,-0.8908027574559069,-0.2397899414842117,-0.2519465767915537,0.7007866975099171,1.0349762822880437,-1.0859321645860662,-0.9185934814924185,
                0.22721180411807995,0.35961283743857103,-0.8412144489775768,0.4228089058048054,0.4476334215888618,0.3680591022537134,-0.1710817296909668,-2.050808285958871,-1.8424125010759516,0.3817969651705484,
                0.34016554148203226,-0.5792926246074874,-0.5786535716115723,-0.7296781397189692,0.6310644859414388,1.4215038696200402,-0.1673884792728171,0.3663686531315469,0.20192140106374845,-1.9501146753730902,
                -2.3523665140196246,0.6211263746798362,-1.0704706894881901,-0.9359135417860172,-0.15538958215467039,0.8492603661088168,0.6985269364947899,0.2543737984367715,0.5900819119114372,-0.46586534286543335,
                0.646272474818845,1.5584981303532475,-1.8169819418259323,-0.10808360019685824,0.3039235818922141,0.38588707934407906,-1.2071409817909105,-1.5117475152502058,0.7865117326708584,-1.0433610183881474,
                -0.022887885914118965,0.537696986266523,0.16194014874682913,0.8774008703179472,-0.3357473198057517,2.297553492563504,0.7884373509103298,-0.9910054154833914,-1.8161476913203296,-1.5442040758088555,
                -0.6455828587606391,0.5126281889764698,-0.734388633743214,1.1163458531256725,-1.0085229838881533,-0.7676264809047959,1.4018746136362663,0.2102466243799568,-1.572119121430016,-0.6739316718793418,
                1.753999496488195,0.6832006039339151,0.8230323618546288,1.185199678571156,1.5095906455072448,-1.8879231512148054,0.812529865605874,0.8914589399893891,0.10703919707456588,-2.329754414364913,
                -0.32120137758273193,-0.23502047184911834,0.16697827217002315,0.20995751357316678,-0.9716168223307883,0.982474447089885,-0.40373222491726685,-0.8318783048967151,-1.7358284888483095,0.40206580471232234,
                1.503951607313408,-0.47854948855358603,0.1964748483822598,2.51365761778254,1.3593058564507452,0.8459471341258273,0.23780360683481822,-1.48080209396047,-2.075411247751218,1.856996475612273,
                -0.23135027224834687,-0.1006179151592303,0.3215400180128015,-0.3362515478608336,-0.249607992786543,-0.3111882282039243,0.19078550843055084,-2.188623277422953,-0.036115314782217224,-0.5839692472759664,
                -0.20556497750706712,-0.4214341128933532,0.0324438084709395,1.4279553742226918,0.8173480978624985,1.0581861114915185,1.8916941011595667,1.1899283044356892,-0.7926496722222311,-0.755113974583242,
                -0.353301849462259,-0.5030673870388486,1.6286743040589062,0.7177006874866977,1.4969075213632772,0.2638381388352704,-1.675347897520038,-1.5333331890344686,-1.2093503657491125,0.030198011172445718,
                0.6021094415678825,-1.5599350791235813,0.3431017597202864,1.2824942677177993,-0.05433098495981555,-0.07843313659948174,-0.016307326327780567,0.6067925716490657,0.7294361755078453,0.6054475986074143);

        network.setW2(w2);*/



    }


    @Test
    public void twoLayerNet한줄짜리() throws Exception{

        J06_TwoLayerNet network = new J06_TwoLayerNet(15, 20, 10, 0.1f);

        Matrix w1 = Matrix.create(15, 20);
        w1.setElements(
                -0.5832131285233637,0.5546232317776113,-1.02258247420341,0.8243164440349637,-0.16639924876385725,-0.23332879890749791,-0.6978080401320819,-1.201138393353154,0.6678297489224253,-0.12098475252919609,-0.3636192820944814,-0.007116726710625545,0.5851253358592502,-0.06486206734020675,0.15462802416520918,1.2560328814710902,1.6170342206619672,-0.7247052795621695,-1.6738737773367534,0.11485358844937661,
                -2.1960044074720426,0.10916767641474101,-0.8575282080336422,0.18146622860526446,1.0230975164803955,2.230546314861665,0.6839562589307516,1.2765940659157402,0.6400894137026407,-0.5025554862735423,-0.7830607361970826,0.1399844182355591,0.5170155683282845,-0.4582405779393589,-0.6861103238026324,0.5384687990745262,-0.19657543178713882,0.8692159599602527,-0.06879590497139987,2.0016463211672337,
                -2.0390003112225257,-0.13501383380737403,0.092551957028291,-1.4301706064265236,-1.51646142626731,0.6527863328651854,-0.4375769095516733,-0.2616571060094534,-1.0181558623756053,0.8993809768911969,-0.0755142168431911,2.7958139075404085,-0.9579814507707247,1.5728399659861492,0.3606992172085906,0.5611924868436017,-0.554841220868448,-0.9154580477737951,1.15334814851474,0.5546185212756523,
                0.06355656334207033,-0.1564326902667807,-1.1779297458376583,0.5123709097156922,1.2344112683144597,1.486150566841667,-0.2656485208299998,-0.6582627462826944,1.2414377206727925,3.295685602219065,-0.34442274821358493,0.26789653970573846,1.0101781657401556,-0.4444460974875524,0.41878758441990216,-0.42114343670862553,-0.5039326112856248,-1.1175033451463916,-1.3255982346040183,0.8664582941259302,
                -0.9981697979694187,-0.6117009119704756,-0.8404951759143134,-0.05026642682866235,-1.392078183064078,0.38148123098229203,0.8731303871142118,-0.7971914854777031,0.037434888004452334,1.2263118234320207,-0.08686556595731332,0.07037005026753147,0.0021901832697430327,0.4363685258130466,0.7932908094622271,-0.9709463489578078,-0.5265425260318936,-0.3107699026718654,-0.12722559128549707,0.38789011619014235,
                1.7057266489173435,-0.10498255158641262,-0.7024631154258564,-0.5574046064220263,-0.4297167073684931,0.5447157609910309,0.6692689663209391,0.09165290254326851,-0.7149292707115317,0.2603879368800421,-0.14190335154359446,0.9444251008988646,-0.16289310532216156,0.7936831446227277,1.9076124162161536,1.2560071131967034,0.4744283664898206,-0.6373079311249463,-0.8862674144658899,1.1488219548262018,
                -0.7071055767788609,-1.1722219757010766,1.0641065357233566,-1.0856967496886074,0.5429434057351684,1.046131644746873,-0.48173132441914723,0.3005620147758014,0.7057986316540363,-0.339910699358982,3.0566982225170105,-1.1009172379738819,-0.35876979780913637,0.6390452190446086,0.5904786177572018,0.7967508041427749,0.30757203102045083,-0.7033609011941422,-1.3138617559610963,-0.5593599700369796,
                0.4210924757104555,0.5055570506074062,-0.6109924875934407,-1.3775663862041319,0.007036575484356437,-0.11078080511459974,0.8874614574689242,-1.2166706854410587,-0.5967172456437143,0.3641945803867415,1.1460458902828468,0.7316743555095526,1.3150046223118066,-0.5917006384234611,0.8816146520994863,0.5422387431049939,-0.004976675311718363,0.8549470780793487,-1.321977948150516,1.136582567945368,
                -0.9426481910237337,-0.8895824838760034,-0.6114599642573931,-0.9623695478063362,-0.23925517989959552,-0.5763197196534189,0.8883582229249963,1.8694276283026605,-0.2495829032138117,0.29850782075128823,-0.425353928907089,-0.19061913969037517,-1.128683213786236,2.048653254224502,1.2826827077902903,0.42633173071407693,-0.7354438233972094,0.1988099290524459,-0.428854584751412,0.018013708467837026,
                0.4099147123921482,-0.700213276624436,-1.044011330283384,-1.0049743880863193,-0.437571102390899,-0.8989489351179052,-0.3056081352582339,-0.5690269892346571,0.3853013121274834,-0.12904594802647196,0.32339448957323447,-0.40319001539888866,0.5501092636486908,-0.9265274156474073,-0.9802672252798936,0.41473532285996273,0.07826652971402591,-0.454868297919447,1.8024070792797193,-0.6573692015735509,
                1.936854734807018,-0.3781629013357878,0.6095258501379681,-2.027535207909814,0.4922727131072454,0.24549150596211372,0.9199296503290153,0.8337851991941454,0.527289442211607,0.6146670617060478,0.23690266241594118,0.5282036255654071,0.13292709477553422,-0.5319134793419927,-0.20113752691845013,-1.2866615494236788,-2.6220125848642977,-2.7468100518395353,0.9766853981755453,-0.009892207883805943,
                -0.1511812396455604,-0.19288905897904038,1.278265816424119,-0.4452538024937409,-1.0492667077407856,-0.8082958257085043,0.6049255035175471,-0.08407689873698312,0.39263718868780767,-0.36498941516890554,-0.2779422154431738,-1.027564700877942,-0.5832604198997365,0.3337681703762059,1.0103495311878004,-0.696813935626284,-1.2747832086814785,0.6334362082219487,-0.5725935547076859,0.8230490599827827,
                -0.8503978311833648,-1.3352599010383273,-0.555867402694812,0.08217368476905423,0.9037175027528601,-1.025760423996414,0.28883918599750336,0.9778129635549985,-1.4010630220738218,-1.6862379663614682,-1.0955029246231962,-0.5986080046647649,0.579509389476287,-0.4782125300852835,1.4138808503170357,-1.3765557119742227,-0.49869184368607733,1.3403371413421925,0.8078583545426257,-1.838747552215498,
                1.317786080681839,0.4726664725466116,0.7497570746023539,0.851423209935642,-0.9428367274638262,0.7330765054896953,1.2508733403013559,-0.3415506461046698,2.0700040005184217,-0.7742063231757103,-0.5718431009661537,-0.42450578264210065,-0.6129012231287408,-0.21521770545157365,-0.9656202368780213,-0.7273831074223376,1.5047988698212382,0.47874828726263546,-0.11939200766086822,-0.7977484889564701,
                -0.07189203708422678,1.5417523712456118,-0.20639314103985382,-0.9816166452965714,-0.9389386813164481,-0.3903318680309627,1.7295111888274988,0.5492312364915635,0.47168936537071837,1.5752578615836186,0.8341794797001169,1.345889874659422,0.8961770839556652,-0.6187424288977845,1.3496788291423907,0.048560565473247226,9.391440357197202E-4,-1.064305461547677,-0.530549677708072,0.6254009088920068
        );
        network.setW1(w1);

        Matrix w2 = Matrix.create(20, 10);
        w2.setElements(0.12832165934351278,1.7307457418865164,-0.7955517175271501,0.19705962940561667,0.5791455628924007,1.2317936856687322,-0.3984270633160795,-1.3066620482926152,-0.5979459246817268,-0.28367295788347885,
                -0.8351354988031502,-0.4234769253133356,0.2081429695448249,0.8124141199983028,-0.21545678533745774,-1.4239417004968398,1.6904424014515442,-0.6605048267651356,-0.3360248271554711,1.855273147505157,
                0.23572016333681994,0.967018393501678,1.9569337442087207,-0.22848795872248798,0.19417730373070852,0.43148779534515014,-1.9290933736540719,-0.33616051270797803,1.1472968153002252,2.567949635577401,
                -0.545833208870957,0.24572747991019186,0.10965603027782918,-0.2227829520110617,-0.07873090964425418,1.7894390905399669,0.7563932147744111,-0.8261058196366828,1.356788314304989,-0.3409166574128076,
                -2.5085485510095906,0.3219407097624265,1.3232893653415583,-1.2238935566837525,-0.8328049685720232,0.3647287105961014,1.271768612671993,0.6375754258988534,-0.9483250619861961,0.7362274782559146,
                -1.3109157181995927,0.615167745143953,-0.6783338683551938,1.1354478501378933,-0.28165165307409507,-0.06741934800698555,0.0340076433696741,0.29271571322516826,-1.5816217846030547,-0.14010275094998617,
                -1.1447375659441485,0.7481451330919384,0.08503131968410195,-0.8908027574559069,-0.2397899414842117,-0.2519465767915537,0.7007866975099171,1.0349762822880437,-1.0859321645860662,-0.9185934814924185,
                0.22721180411807995,0.35961283743857103,-0.8412144489775768,0.4228089058048054,0.4476334215888618,0.3680591022537134,-0.1710817296909668,-2.050808285958871,-1.8424125010759516,0.3817969651705484,
                0.34016554148203226,-0.5792926246074874,-0.5786535716115723,-0.7296781397189692,0.6310644859414388,1.4215038696200402,-0.1673884792728171,0.3663686531315469,0.20192140106374845,-1.9501146753730902,
                -2.3523665140196246,0.6211263746798362,-1.0704706894881901,-0.9359135417860172,-0.15538958215467039,0.8492603661088168,0.6985269364947899,0.2543737984367715,0.5900819119114372,-0.46586534286543335,
                0.646272474818845,1.5584981303532475,-1.8169819418259323,-0.10808360019685824,0.3039235818922141,0.38588707934407906,-1.2071409817909105,-1.5117475152502058,0.7865117326708584,-1.0433610183881474,
                -0.022887885914118965,0.537696986266523,0.16194014874682913,0.8774008703179472,-0.3357473198057517,2.297553492563504,0.7884373509103298,-0.9910054154833914,-1.8161476913203296,-1.5442040758088555,
                -0.6455828587606391,0.5126281889764698,-0.734388633743214,1.1163458531256725,-1.0085229838881533,-0.7676264809047959,1.4018746136362663,0.2102466243799568,-1.572119121430016,-0.6739316718793418,
                1.753999496488195,0.6832006039339151,0.8230323618546288,1.185199678571156,1.5095906455072448,-1.8879231512148054,0.812529865605874,0.8914589399893891,0.10703919707456588,-2.329754414364913,
                -0.32120137758273193,-0.23502047184911834,0.16697827217002315,0.20995751357316678,-0.9716168223307883,0.982474447089885,-0.40373222491726685,-0.8318783048967151,-1.7358284888483095,0.40206580471232234,
                1.503951607313408,-0.47854948855358603,0.1964748483822598,2.51365761778254,1.3593058564507452,0.8459471341258273,0.23780360683481822,-1.48080209396047,-2.075411247751218,1.856996475612273,
                -0.23135027224834687,-0.1006179151592303,0.3215400180128015,-0.3362515478608336,-0.249607992786543,-0.3111882282039243,0.19078550843055084,-2.188623277422953,-0.036115314782217224,-0.5839692472759664,
                -0.20556497750706712,-0.4214341128933532,0.0324438084709395,1.4279553742226918,0.8173480978624985,1.0581861114915185,1.8916941011595667,1.1899283044356892,-0.7926496722222311,-0.755113974583242,
                -0.353301849462259,-0.5030673870388486,1.6286743040589062,0.7177006874866977,1.4969075213632772,0.2638381388352704,-1.675347897520038,-1.5333331890344686,-1.2093503657491125,0.030198011172445718,
                0.6021094415678825,-1.5599350791235813,0.3431017597202864,1.2824942677177993,-0.05433098495981555,-0.07843313659948174,-0.016307326327780567,0.6067925716490657,0.7294361755078453,0.6054475986074143);

        network.setW2(w2);

        // M nist 이미지 가져오기
        LoadMnist loadMnist = new LoadMnist();
        Map<Integer, Matrix> xtrainSampleMap = loadMnist.getXtrainSampleMap();
        Map<Integer, Matrix> ttrainSampleMap = loadMnist.getTtrainSampleMap();

        // 2 세팅
        int size = 1;
        Matrix x_batch = Matrix.create(size, 15);
        Matrix t_batch = Matrix.create(size, 10);
        x_batch.setRow(0, xtrainSampleMap.get(2).getRow(0)); // 2 가져옴
        t_batch.setRow(0, ttrainSampleMap.get(2).getRow(0)); // 2 정답  t 테이블 가져옴

        // Predict
        Matrix predict = network.predict(x_batch);
        log.info("predict : {}", predict);
    }


    @Test
    public void accuracyTest() throws Exception{
        J06_TwoLayerNet network = new J06_TwoLayerNet(10, 10, 10, 0.01f);

        mikera.vectorz.Vector v = mikera.vectorz.Vector.of(1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        Matrix x = Matrix.create(10, 10);
        Matrix t = Matrix.create(10, 10);

        x.add(v);
        t.add(v);

        double accuracy = network.calcul_accracy(x, t);
        assertThat(accuracy).isEqualTo(1.0, offset(0.1));

        t.set(9, 0, 0.0);
        t.set(9, 1, 1.0);
        accuracy = network.calcul_accracy(x, t);
        assertThat(accuracy).isEqualTo(0.9, offset(0.01));
    }

    @Test
    public void addRowTest() throws Exception{
        mikera.vectorz.Vector v = mikera.vectorz.Vector.of(1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        Matrix x = Matrix.create(10, 10);

        x.add(v);

        log.info("v: {}", x);
        x.add(v);

        log.info("v: {}", x);
    }


    @Test
    public void gradTest() throws Exception{

        Matrix x = Matrix.create(3,3);
        x.setElements(
                1,2,3,
                4,5,6,
                7,8,9);

        Matrix t = Matrix.create(3,3);
        t.setElements(
                1,0,0,
                1,0,0,
                1,0,0);


        J06_TwoLayerNet network = new J06_TwoLayerNet(3, 3, 3, 1.0f);
        network.setW1(x);
        network.setW2(x.copy());
        log.info("network w2 :{}", network.getW2());
        Grad grad = network.numerical_gradient(x, t);
        log.info("after grad: \n {}", grad);
        // network.renewParams(grad, 1.0);


        /*grad = network.numerical_gradient(x, t);
        network.renewParams(grad, 1.0);*/

    }






}