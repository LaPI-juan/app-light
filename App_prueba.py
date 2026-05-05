import streamlit as st
import SimpleITK as sitk
import gdown
import tempfile
import os
import requests
import textwrap
from PIL import Image
import numpy as np

#### Funciones propias ####
from RotarVolumen import abrir_archivos_dicom, process_dicom
from inferencia import uso_RUBEN, uso_YOLO, CargarVolumen_YOLO
from conversor import carpetaPNG, carpetaPNG_paths, carpetaDCM

#### Estilo HTML ####

#### Tarjeta #### 
st.markdown('''
        <style>
        .card {
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 12px;
            border: 1px solid #ddd;
            margin-top: 10px;
        }
        .card h4 {
            margin-top: 0;
        }
        </style>
        ''', unsafe_allow_html=True)

#### Inicializar #### 
if 'screen' not in st.session_state:
    st.session_state.screen = 1

def go_welcome():
    st.session_state.screen = 1

def go_lobby():
    st.session_state.screen = 2

#### Presentacion #### 
if st.session_state.screen == 1:
    st.title('Welcome')
    st.markdown('Press **Start** to enter')
    st.button('Start ▶️', on_click = go_lobby)
	
#### Lobby #### 
elif st.session_state.screen == 2:

    st.set_page_config(layout='wide')
	
    upload_dcm_1 = st.sidebar.file_uploader('**Archivos DCM 1**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm1')
    
    upload_dcm_2 = st.sidebar.file_uploader('**Archivos DCM 2**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm2')

    upload_dcm_3 = st.sidebar.file_uploader('**Archivos DCM 3**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm3')

    upload_dcm_4 = st.sidebar.file_uploader('**Archivos DCM 4**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm4')

    upload_dcm_5 = st.sidebar.file_uploader('**Archivos DCM 5**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm5')

    upload_dcm_6 = st.sidebar.file_uploader('**Archivos DCM 6**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm6')
    
    upload_dcm_7 = st.sidebar.file_uploader('**Archivos DCM 7**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm7')

    upload_dcm_8 = st.sidebar.file_uploader('**Archivos DCM 8**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm8')

    upload_dcm_9 = st.sidebar.file_uploader('**Archivos DCM 9**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm9')

    upload_dcm_10 = st.sidebar.file_uploader('**Archivos DCM 10**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm10')

    upload_dcm_11 = st.sidebar.file_uploader('**Archivos DCM 11**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm11')
    
    if upload_dcm_1:
        
        upload_dcms = [upload_dcm_1,upload_dcm_2,upload_dcm_3]#,upload_dcm_4,upload_dcm_5,
                       #upload_dcm_6,upload_dcm_7,upload_dcm_8,upload_dcm_9,upload_dcm_10,
                       #upload_dcm_11]
        
        #st.write(len(upload_dcms))
        # ------------------------------------------------------------------------------------
        #                                   SUBIDA DE DATOS
        # ------------------------------------------------------------------------------------
		#### Carpeta temporal DICOM ####
        if 'rutas_DCM' not in st.session_state:
            rutas_DCM = []

            for upload_dcm in upload_dcms:
                temp_dcm_org = tempfile.mkdtemp()
                
                for f in upload_dcm:
                    path = os.path.join(temp_dcm_org, f.name)
                    with open(path, 'wb') as out:
                        out.write(f.getbuffer())

                rutas_DCM.append(temp_dcm_org)

            st.session_state.rutas_DCM = rutas_DCM
            st.session_state.NV = len(rutas_DCM)

        rutas_DCM = st.session_state.rutas_DCM
        NV = st.session_state.NV
		
        # ------------------------------------------------------------------------------------
        #                                      ORIGINAL
        # ------------------------------------------------------------------------------------
		#### Carpeta temporal PNG ####
        if 'temp_png_orgs' not in st.session_state:
            temp_png_orgs = []
            for ruta_DCM in rutas_DCM:
                V = abrir_archivos_dicom(ruta_DCM)
                sld_nums = V.shape[0:2]
                temp_png_org = carpetaPNG(V,0)
                temp_png_orgs.append(temp_png_org)

            st.session_state.temp_png_orgs = temp_png_orgs
            st.session_state.sld_nums = sld_nums

        temp_png_orgs = st.session_state.temp_png_orgs
        sld_nums = st.session_state.sld_nums

        # ------------------------------------------------------------------------------------
        #                                   ESTÁNDAR
        # ------------------------------------------------------------------------------------
        #### Inferencia de los parámetros ####
        if 'p_std' not in st.session_state: 
            url_estandar = 'https://drive.google.com/uc?export=download&id=10BglUsjZLKeeiqpGG5xY-Isy8vGAzrNg'
            output_estandar = 'modelo_estandar.pt'
            gdown.download(url_estandar, output_estandar, quiet=False)
            st.session_state.p_std = uso_RUBEN('modelo_estandar.pt',
											   temp_png_orgs[0])
        
        p_std = st.session_state.p_std

		#### Carpetas temporales ####
        if 'temp_png_stds' not in st.session_state:
            temp_dcm_stds, temp_png_stds = [], []
            for ruta_DCM in rutas_DCM: 
                V, spc_std = process_dicom(p_std[0], p_std[1:4], ruta_DCM)
                temp_dcm_std = carpetaDCM(V, spc_std)
                temp_png_std = carpetaPNG(V,0)
                temp_dcm_stds.append(temp_dcm_std)
                temp_png_stds.append(temp_png_std)
				
            st.session_state.temp_dcm_stds = temp_dcm_stds
            st.session_state.temp_png_stds = temp_png_stds
			
        temp_dcm_stds = st.session_state.temp_dcm_stds
        temp_png_stds = st.session_state.temp_png_stds
        
        # ------------------------------------------------------------------------------------
        #                                         LVOT
        # ------------------------------------------------------------------------------------
        #### Inferencia de los parámetros ####
        if 'p_LVOT' not in st.session_state:
            url_LVOT = 'https://drive.google.com/uc?export=download&id=15g-yPz4mVrDpgN4h0tnuuE7dt7-ZxRj_'
            output_LVOT = 'modelo_LVOT.pt'
            gdown.download(url_LVOT, output_LVOT, quiet=False)
            st.session_state.p_LVOT = uso_RUBEN('modelo_LVOT.pt',
												temp_png_stds[0])
        
        p_LVOT = st.session_state.p_LVOT

    	#### Carpetas temporales ####
        if 'temp_png_LVOTs' not in st.session_state:
            temp_png_valvs, temp_png_LVOTs = [], []
            for temp_dcm_std in temp_dcm_stds:
                V, _ = process_dicom(p_LVOT[0], p_LVOT[1:4], temp_dcm_std)
                temp_png_LVOT = carpetaPNG(V,0)
                temp_png_LVOTs.append(temp_png_LVOT)

                V = V.transpose(1,0,2)
                temp_png_valv = [carpetaPNG(V[np.newaxis,i,:,:],0) for i in range(V.shape[0])]
                temp_png_valvs.append(temp_png_valv)

            st.session_state. temp_png_LVOTs = temp_png_LVOTs
            st.session_state.temp_png_valvs = temp_png_valvs
            
        temp_png_LVOTs = st.session_state.temp_png_LVOTs
        temp_png_valvs = st.session_state.temp_png_valvs
		
        # ------------------------------------------------------------------------------------
        #                                    YOLO
        # ------------------------------------------------------------------------------------
        if 'temp_png_YOLOs' not in st.session_state:
            temp_png_YOLOs = []
            for temp_png_valv in temp_png_valvs:
                temp_png_YOLO = []
                for ruta in temp_png_valv:
                    V = CargarVolumen_YOLO(ruta)
                    img_YOLO = carpetaPNG(V[:,:,:,0],0)
                    temp_png_YOLO.append(img_YOLO)
                temp_png_YOLOs.append(temp_png_YOLO)

            st.session_state.temp_png_YOLOs = temp_png_YOLOs
			
        temp_png_YOLOs = st.session_state.temp_png_YOLOs

        tab1, tab2, tab3 = st.tabs(['Estándar', 'LVOT', 'Mascara'])

        # ------------------------------------------------------------------------------------
        #                                   PESTAÑA VISTA ESTANDAR
        # ------------------------------------------------------------------------------------
        with tab1:
	        html_3 = f'''
    	        <div class="card">
        	        <h4>Imagenes</h4>
            	</div>
        	'''
        	st.markdown(textwrap.dedent(html_3), unsafe_allow_html=True)
        
        	N_org_1 = st.slider('Volumen',min_value=1, max_value=NV, step=1,key ='N_org1')
        	N_fnl_1 = st.slider('Corte',min_value=1, max_value=sld_nums[0], step=1,key ='N_fnl1')

        	img_orig_user_1 = Image.open(os.path.join(temp_png_orgs[N_org_1-1], f'slice_{(N_fnl_1-1):03d}.png'))
        	img_fnl_user_1 = Image.open(os.path.join(temp_png_stds[N_org_1-1], f'slice_{(N_fnl_1-1):03d}.png'))

        	col1, col2 = st.columns(2)
        	with col1:
        	    st.image(img_orig_user_1, caption='Original', use_container_width=True)
        	with col2:
        	    st.image(img_fnl_user_1, caption='Estandar', use_container_width=True)

        # ------------------------------------------------------------------------------------
        #                                   PESTAÑA VISTA LVOT
        # ------------------------------------------------------------------------------------
        with tab2:
	        html_3 = f'''
    	        <div class="card">
        	        <h4>Imagenes</h4>
            	</div>
        	'''
        	st.markdown(textwrap.dedent(html_3), unsafe_allow_html=True)
        
        	N_org_2 = st.slider('Volumen',min_value=1, max_value=NV, step=1,key ='N_org2')
        	N_fnl_2 = st.slider('Corte',min_value=1, max_value=sld_nums[0], step=1,key ='N_fnl2')

        	img_orig_user_2 = Image.open(os.path.join(temp_png_stds[N_org_2-1], f'slice_{(N_fnl_2-1):03d}.png'))
        	img_fnl_user_2 = Image.open(os.path.join(temp_png_LVOTs[N_org_2-1], f'slice_{(N_fnl_2-1):03d}.png'))

        	col1, col2 = st.columns(2)
        	with col1:
        	    st.image(img_orig_user_2, caption='Original', use_container_width=True)
        	with col2:
        	    st.image(img_fnl_user_2, caption='Estandar', use_container_width=True)

        # ------------------------------------------------------------------------------------
        #                                   PESTAÑA DETECCION
        # ------------------------------------------------------------------------------------
        with tab3:
            N_org_3 = st.slider('',min_value=1, max_value=NV, step=1, key = 'sld_valva_1')
            N_fnl_3 = st.slider('',min_value=1, max_value=sld_nums[1], step=1, key = 'sld_valva_2')

            img_fnl_user_3 = Image.open(os.path.join(temp_png_YOLOs[N_org_3-1][N_fnl_3-1],'slice_000.png'))

            valv1, valv2, valv3 = st.columns([1,2,1])
            with valv1:
                st.write('')

            with valv2:
                st.image(img_fnl_user_3, caption=' ', use_container_width=True)

            with valv3:
                st.write('')
