1. SDK_Python����˵����
   �����Ѿ��ṩ�˱�ķʽ�ķ�����ֻ��Ҫ����
   ���predictor.py�ļ��е�XXX������
   SDK�Ѿ�ʵ���˶�ȡ�ļ�����Ҫ���ʽд�ļ��Լ���ӡ��ʼ�ͽ���ʱ��Ĺ��ܡ�Ϊ�˱��ڵ��ԣ�SDK���������Ϣȫ������Ļ������ɸ����������Ҫ����ɾ���˴�ӡ��Ϣ��
   ע�⣺��ȡ�ļ�������ָ����ͼ����Ϣ�ļ���·����Ϣ�ļ����ж�ȡ���ڴ棬�����ڴ��еĴ洢��ʽ�����ַ�����ʽ����Ϊ��Щ��Ϣ��ʲô��ʽ�洢�漰���㷨��ƣ���������Ϊ�˲��������˼·��

2. Python �汾Ҫ��Ϊ�� python 2.7.5

3. ����ʱֻ��Ҫִ�У� python ecs.py ../TrainData.txt ../input.txt ../output.txt
	(˵����TrainData.txt����ʷ�����ļ���input.txt���������������ļ���output.txt������ļ�)

4. ����ϴ��ļ���Ҫ�����ļ�·����
   src/ecs/ecs.py
   src/ecs/predictor.py
   �������Լ������py�ļ�Ҳ����ӣ� ��֤�����õõ�����ֱ���޸�SDK_PythonΪ�Լ����֣����tar.gz���ϴ����ɣ���zhangsan.tar.gz��
  
  
  python ecs.py ../test_file/TrainData.txt ../test_file/input_5flavors_cpu_7days.txt ../test_file/output.txt
  
  python ecs.py ../test_file/data_2015_1.txt ../test_file/input_5flavors_cpu_7days.txt ../test_file/output.txt
  
  python ecs.py ../test_file/data_2015_2.txt ../test_file/input_5flavors_cpu_7days.txt ../test_file/output.txt
  
  python ecs.py ../test_file/data_2016_1.txt ../test_file/input_15flavors_cpu_7days.txt ../test_file/output.txt
  