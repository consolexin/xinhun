```java
import java.io.File;
import java.io.FileInputStream;
import java.util.Arrays;
import java.util.Random;

public class AI3 {
	/**
	 * 输入层神经元个数N<br>对应i
	 */
	public static int N=9080;
	/**
	 * 中间层神经元个数L<br>对应j
	 */
	public static int L=100;
	/**
	 * 输出层神经元个数M<br>对应k
	 */
	public static int M=15;
	/**
	 * 输入图片的字节数组
	 */
	public static byte[] INPUTBYTE=new byte[N];
	/**
	 * 输入图片的浮点数组
	 */
	public static double[] INPUT=new double[N];
	
	/**
	 * 输入单元到中间单元的权值
	 */
	public static double[][] V=new double[N][L];
	/**
	 * 中间层的阈值
	 */
	public static double[] phi=new double[L];
	/**
	 * 中间层的阈值调整梯度
	 */
	public static double[] Deltaphi=new double[L];
	/**
	 * 中间层的误差项
	 */
	public static double[] deltaStar=new double[L];
	/**
	 * 输入单元到中间单元的权值调整的梯度
	 */
	public static double[][] DeltaV=new double[N][L];
	/**
	 * 中间层输出
	 */
	public static double[] h=new double[L];
	
	
	/**
	 * 中间单元到输出层的权值
	 */
	public static double[][] W=new double[L][M]; 
	/**
	 * 输出层的阈值
	 */
	public static double[] theta=new double[M];
	/**
	 * 输出层的阈值调整的梯度
	 */
	public static double[] Deltatheta=new double[M];
	/**
	 * 输出层的误差项
	 */
	public static double[] delta=new double[M];
	/**
	 * 中间单元到输出层的权值调整梯度
	 */
	public static double[][] DeltaW=new double[L][M];
	/**
	 * 实际输出
	 */
	public static double[] y=new double[15];
	/**
	 * 期望输出
	 */
	public static double[] d=new double[15];
	/**
	 * 精度控制参数
	 */
	public static double varepsilon=0.01;
	/**
	 * 学习因子
	 */
	public static double alpha=0.3;
	public static Random rand=new Random();
	/**
	 * 激励函数
	 * @return
	 */
	public static double f(double x) {
		return 1.0/(1.0+Math.pow(Math.E, -x));
	}
	/**
	 * 初始化各权值和阈值
	 */
	public static void init() {
		for(int i=0;i<V.length;i++) {
			for(int j=0;j<V[i].length;j++) {
				V[i][j]=rand.nextDouble()*2-1;
			}
		}
		for(int i=0;i<W.length;i++) {
			for(int j=0;j<W[i].length;j++) {
				W[i][j]=rand.nextDouble()*2-1;
			}
		}
		for(int i=0;i<phi.length;i++) {
			phi[i]=rand.nextDouble()*2-1;
		}
		for(int i=0;i<theta.length;i++) {
			theta[i]=rand.nextDouble()*2-1;
		}
	}
	/**
	 * 计算中间层和输出层的输出
	 * p代表第p个人
	 */
	public static void CalcDHY(int p) {
		for(int i=0;i<15;i++) {
			d[i]=0.0;
		}
		d[p]=1.0;
//		System.out.println(Arrays.toString(d));
		for(int j=0;j<h.length;j++) {
			double m1=0.0;
			for(int i=0;i<N;i++) {
				m1+=INPUT[i]*V[i][j];
			}
			m1+=phi[j];
			h[j]=f(m1);
		}
//		System.out.println("h="+Arrays.toString(h));
		for(int k=0;k<M;k++) {
			double m2=0.0;
			for(int j=0;j<L;j++) {
				m2+=h[j]*W[j][k];
			}
			m2+=theta[k];
			y[k]=f(m2);
		}
//		System.out.println("y="+Arrays.toString(y));
	}
	/**
	 * 计算中间层和输出层的误差项
	 */
	public static void CalcError() {
		for(int k=0;k<M;k++) {
			delta[k]=(d[k]-y[k])*y[k]*(1-y[k]);
		}
		for(int j=0;j<L;j++) {
			double m3=0.0;
			for(int k=0;k<M;k++) {
				m3+=delta[k]*W[j][k];
			}
			deltaStar[j]=h[j]*(1-h[j])*m3;
		}
	}
	/**
	 * 调整权值和阈值
	 */
	public static void reviseVW() {
		for(int j=0;j<L;j++) {
			for(int k=0;k<M;k++) {
				DeltaW[j][k]=(alpha/(1.0+L))*(DeltaW[j][k]+1)*delta[k]*h[j];
			}
		}
		for(int i=0;i<N;i++) {
			for(int j=0;j<L;j++) {
				DeltaV[i][j]=(alpha/(1.0+N))*(DeltaV[i][j]+1)*deltaStar[j]*INPUT[i];
			}
		}
		for(int k=0;k<M;k++) {
			Deltatheta[k]=(alpha/(1.0+L))*(Deltatheta[k]+1)*delta[k];
		}
		for(int j=0;j<L;j++) {
			Deltaphi[j]=(alpha/(1.0+L))*(Deltaphi[j]+1)*deltaStar[j];
		}
		
		
		for(int j=0;j<L;j++) {
			for(int k=0;k<M;k++) {
				W[j][k]+=DeltaW[j][k];
			}
		}
		for(int i=0;i<N;i++) {
			for(int j=0;j<L;j++) {
				V[i][j]+=DeltaV[i][j];
			}
		}
		for(int k=0;k<M;k++) {
			theta[k]+=Deltatheta[k];
		}
		for(int j=0;j<L;j++) {
			phi[j]+=Deltaphi[j];
		}
	}
	/**
	 * 计算误差精度
	 */
	public static void CalcE() {
		@SuppressWarnings("unused")
		double E=0.0;
		for(int k=0;k<M;k++) {
			E+=(d[k]-y[k])*(d[k]-y[k]);
		}
		E/=2.0;
//		System.out.println("本次训练的误差为"+E);
	}
	/**
	 * 根据最后的权值和阈值输出结果
	 */
	public static void output() {
		for(int i=0;i<N;i++) {
			for(int j=0;j<L;j++) {
				System.out.print("Vij="+V[i][j]);
			}
			System.out.println();
		}
		System.out.println("------------------------");
		for(int j=0;j<L;j++) {
			for(int k=0;k<M;k++) {
				System.out.print("Wjk="+W[j][k]);
			}
			System.out.println();
		}
		System.out.println("------------------------");
		for(int j=0;j<L;j++) {
			System.out.print("\tphij="+phi[j]);
		}
		System.out.println("------------------------");
		for(int k=0;k<M;k++) {
			System.out.print("\tthetak="+theta[k]);
		}
		System.out.println("------------------------");
	}
	/**
	 * 找到是第几个人
	 */
	public static void FindMAX() {
		int max_d=0;
		for(int k=1;k<M;k++) {
			if (d[k]>d[max_d]) {
				max_d=k;
			}
		}
		System.out.printf("这实际上是第%d个人\n",(max_d+1));
		int max_y=0;
		for(int k=1;k<M;k++) {
			if (y[k]>y[max_y]) {
				max_y=k;
			}
		}
		System.out.printf("神经网络识别出这是第%d个人\n",(max_y+1));
	}
	public static void main(String[] args) throws Exception{
		init();	
//		String trainPath="D:\\train";
		String trainPath=System.getProperty("user.dir")+"./train";
		File file=new File(trainPath);
		File files[]=file.listFiles();
		FileInputStream train=null;
		for(int time=0;time<2;time++) {
			for (int p = 0; p < files.length; p++) {
				train=new FileInputStream(files[p]);
				train.read(INPUTBYTE);
				for(int i=0;i<N;i++) {
					INPUT[i]=(double)(INPUTBYTE[i]+128)/255.0;
				}
				CalcDHY(p/5);
				CalcError();
				reviseVW();
				CalcE();
				
				train.close();
			}
		}
		//output();
//		String testPath="D:\\test";
		String testPath=System.getProperty("user.dir")+"./test";
		File file2=new File(testPath);
		File tests[]=file2.listFiles();
		FileInputStream test = null;
		for(int p=0;p<tests.length;p++) {
			test=new FileInputStream(tests[p]);
			test.read(INPUTBYTE);
			for(int i=0;i<N;i++) {
				INPUT[i]=(double)(INPUTBYTE[i]+128)/255.0;
			}
			CalcDHY(p/6);
			FindMAX();
			System.out.println(Arrays.toString(y));
		}
		
		
//		System.out.println(Arrays.toString(y));
		
		test.close();
	}

}

```
