import bpy
import os
import json


from .global_properties import GlobalProterties
from .logic_name import LogicName


# 全局配置类，使用字段默认为全局可访问的唯一静态变量的特性，来实现全局变量
class GlobalConfig:
    # 全局静态变量,任何地方访问到的值都是唯一的
    gamename = ""
    workspacename = ""
    ssmtlocation = ""
    current_game_migoto_folder = ""
    logic_name = ""
    # 独立模式(未安装SSMT4)下使用的游戏预设，默认EFMI(终末地)
    # 注意此值为类级别存储，read_from_main_json_ssmt4 反复调用时不会被清空覆盖
    standalone_logic_name = LogicName.EFMI
    # 强制使用独立预设开关，为True时独立预设优先于SSMT4的gamePreset
    # 同样为类级别存储，面板绘制反复调用read_from_main_json_ssmt4时不会被清空，
    # 且注册阶段/受限上下文(无法访问场景属性)下也能安全读取
    force_standalone_preset = False
    # 独立预设提示是否已打印过，避免面板每次重绘都刷屏
    _standalone_hint_printed = False
    # 旧版根目录数据迁移提示是否已打印过 (自定义目录数据现存放于其 workplace 子文件夹)
    _workplace_migrate_hint_printed = False
    _main_settings_cache = {}
    _game_settings_cache = {}
    _main_json_mtime = None
    _game_config_json_mtime = None
    _game_config_json_path = ""

    @classmethod
    def is_ssmt4_available(cls) -> bool:
        '''
        检测SSMT4的settings.json是否存在，不存在则视为独立模式运行
        '''
        try:
            return cls._safe_getmtime(cls.path_main_json_ssmt4()) is not None
        except Exception:
            return False

    @classmethod
    def set_standalone_logic_name(cls, logic_name: str):
        '''
        设置独立模式游戏预设，并在SSMT4配置链无法提供logic_name时立即生效
        '''
        cls.standalone_logic_name = logic_name or LogicName.EFMI
        cls._apply_standalone_fallback()

    @classmethod
    def get_standalone_logic_name(cls) -> str:
        '''
        获取独立模式游戏预设，优先读取场景属性(可随.blend文件保存)，
        无法访问场景时(注册阶段/子进程)使用类级别存储的默认值
        '''
        try:
            preset = GlobalProterties.standalone_game_preset()
            if preset:
                cls.standalone_logic_name = preset
        except Exception:
            pass
        return cls.standalone_logic_name or LogicName.EFMI

    @classmethod
    def set_force_standalone_preset(cls, force: bool):
        '''
        设置强制使用独立预设开关，并立即重新解析logic_name
        关闭时会从SSMT4游戏配置缓存中恢复gamePreset，避免残留独立预设值
        '''
        cls.force_standalone_preset = bool(force)
        if not cls.force_standalone_preset:
            # 关闭强制后恢复SSMT4解析出的gamePreset(可能为空)
            cls.logic_name = cls._game_settings_cache.get("gamePreset", "")
        cls._apply_standalone_fallback()

    @classmethod
    def get_force_standalone_preset(cls) -> bool:
        '''
        获取强制使用独立预设开关，优先读取场景属性(可随.blend文件保存)，
        无法访问场景时(注册阶段/子进程)使用类级别存储的值
        '''
        try:
            cls.force_standalone_preset = bool(GlobalProterties.force_standalone_preset())
        except Exception:
            pass
        return cls.force_standalone_preset

    @classmethod
    def _apply_standalone_fallback(cls):
        '''
        logic_name解析优先级：
        1. 开启"强制使用独立预设"时，独立预设始终生效(即使SSMT4解析出了有效gamePreset)
        2. SSMT4配置链解析出非空gamePreset时，使用SSMT4的值(原有行为)
        3. 未安装SSMT4(独立模式)时，回退到独立模式游戏预设
        4. SSMT4存在但未能解析出gamePreset(配置不完整)时，保持为空，
           让导出流程的未知逻辑检查大声报错，而不是静默套用独立预设
        '''
        # 先捕获类级别旧值再刷新场景属性，检测"强制"开关从开到关的切换
        # (加载另一个.blend文件不会触发属性update回调，mtime缓存又会跳过重新解析，
        #  不在此恢复的话会残留上一个文件强制设置的logic_name)
        prev_force = cls.force_standalone_preset
        force = cls.get_force_standalone_preset()
        if prev_force and not force:
            cls.logic_name = cls._game_settings_cache.get("gamePreset", "")

        if force:
            cls.logic_name = cls.get_standalone_logic_name()
            cls._standalone_hint_printed = False
            return

        if cls.logic_name:
            cls._standalone_hint_printed = False
            return

        if not cls.is_ssmt4_available():
            cls.logic_name = cls.get_standalone_logic_name()
            cls._standalone_hint_printed = False
            return

        # SSMT4存在但配置不完整(CurrentGameName为空或游戏Config.json缺失/无gamePreset)
        # 此时不套用独立预设，保持为空让后续流程报错提示；只打印一次控制台提示避免刷屏
        if not cls._standalone_hint_printed:
            print("GlobalConfig: 检测到SSMT4但未能解析出游戏预设(gamePreset)，请检查SSMT4配置；若想改用LoyalTools独立预设，可在基础信息面板开启\"强制使用独立预设\"")
            cls._standalone_hint_printed = True

    @classmethod
    def _safe_getmtime(cls, file_path: str):
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return None

    @classmethod
    def _read_json_file(cls, file_path: str):
        with open(file_path, encoding="utf-8") as json_file:
            return json.load(json_file)

    @classmethod
    def _clear_main_settings(cls):
        cls._main_settings_cache = {}
        cls._main_json_mtime = None
        cls.workspacename = ""
        cls.gamename = ""
        cls.ssmtlocation = ""

    @classmethod
    def _clear_game_settings(cls, game_config_json_path: str = ""):
        cls._game_settings_cache = {}
        cls._game_config_json_mtime = None
        cls._game_config_json_path = game_config_json_path
        cls.current_game_migoto_folder = ""
        cls.logic_name = ""

    @classmethod
    def read_from_main_json_ssmt4(cls) :
        try:
            main_json_path = cls.path_main_json_ssmt4()
            main_json_mtime = cls._safe_getmtime(main_json_path)

            if main_json_mtime is None:
                cls._clear_main_settings()
                cls._clear_game_settings()
                return

            if cls._main_json_mtime != main_json_mtime:
                main_setting_json = cls._read_json_file(main_json_path)
                cls._main_settings_cache = main_setting_json
                cls._main_json_mtime = main_json_mtime
                cls.workspacename = main_setting_json.get("CurrentWorkSpace", "")
                cls.gamename = main_setting_json.get("CurrentGameName", "")

                base_folder = (
                    main_setting_json.get("SSMTWorkFolder")
                    or main_setting_json.get("DBMTWorkFolder", "")
                )
                cls.ssmtlocation = base_folder + "\\" if base_folder else ""

            game_config_json_path = os.path.join(
                cls.path_ssmt4_global_configs_folder(),
                "Games\\" + cls.gamename + "\\Config.json",
            )

            if not cls.gamename:
                cls._clear_game_settings(game_config_json_path)
                return

            game_config_json_mtime = cls._safe_getmtime(game_config_json_path)
            if game_config_json_mtime is None:
                cls._clear_game_settings(game_config_json_path)
                return

            if (
                cls._game_config_json_path != game_config_json_path
                or cls._game_config_json_mtime != game_config_json_mtime
            ):
                game_config_json = cls._read_json_file(game_config_json_path)
                cls._game_settings_cache = game_config_json
                cls._game_config_json_path = game_config_json_path
                cls._game_config_json_mtime = game_config_json_mtime
                cls.current_game_migoto_folder = game_config_json.get("installDir", "")
                cls.logic_name = game_config_json.get("gamePreset", "")
        except Exception as e:
            print(e)
        finally:
            # 独立模式回退：SSMT4配置链未能解析出logic_name时使用独立模式游戏预设
            # 放在finally中确保所有return/异常路径都能生效，且不会覆盖已解析出的值
            cls._apply_standalone_fallback()
            
    @classmethod
    def base_path(cls):
        return cls.ssmtlocation
    
    @classmethod
    def path_drawib_config_json_path(cls):
        '''
        当前工作空间目录下的Config.json
        存储了所有的DrawIB和别名
        '''
        game_config_json_path = os.path.join(GlobalConfig.path_workspace_folder(),"Config.json")
        return game_config_json_path
    
    @classmethod
    def path_configs_folder(cls):
        return os.path.join(GlobalConfig.base_path(),"Configs\\")
    
    @classmethod
    def path_reverse_output_folder(cls):
        cls.read_from_main_json_ssmt4()
        reverse_output_folder = cls._main_settings_cache.get("ReverseOutputFolder", "")
        return reverse_output_folder + "\\" if reverse_output_folder else ""

    @classmethod
    def path_mods_folder(cls):
        return os.path.join(cls.current_game_migoto_folder,"Mods\\") 

    @classmethod
    def path_total_workspace_folder(cls):
        return os.path.join(GlobalConfig.base_path(),"WorkSpace\\") 
    
    @classmethod
    def path_current_game_total_workspace_folder(cls):
        return os.path.join(GlobalConfig.path_total_workspace_folder(),GlobalConfig.gamename + "\\") 

    @classmethod
    def _normalize_workspace_folder_path(cls, folder_path: str) -> str:
        normalized = str(folder_path or "").strip()
        if not normalized:
            return ""

        normalized = os.path.normpath(normalized)
        if not normalized.endswith("\\"):
            normalized = normalized + "\\"
        return normalized

    @classmethod
    def _resolve_custom_workspace_data_folder(cls, custom_workspace_folder_path: str) -> str:
        '''
        自定义目录下的实际数据目录为 <自定义目录>/workplace/
        (提取生成的子网格文件夹与 Import.json / Config.json 都存放在这里，
        保持自定义目录本身整洁)。目录由提取流程按需创建，这里只负责拼路径。
        检测到旧版数据直接放在自定义目录根下时打印一次迁移提示。
        '''
        workplace_folder_path = os.path.join(custom_workspace_folder_path, "workplace\\")
        try:
            if (not os.path.isdir(workplace_folder_path)
                    and os.path.isfile(os.path.join(custom_workspace_folder_path, "Import.json"))
                    and not cls._workplace_migrate_hint_printed):
                print("GlobalConfig: 检测到旧版提取数据位于自定义目录根下 ("
                      + custom_workspace_folder_path
                      + ")，新版数据目录为其中的 workplace 子文件夹，请重新提取或手动把子网格文件夹与 Import.json/Config.json 移入 workplace。")
                cls._workplace_migrate_hint_printed = True
        except Exception:
            pass
        return workplace_folder_path

    @classmethod
    def get_workspace_name(cls):
        try:
            workspace_source_mode = GlobalProterties.workspace_source_mode()

            if workspace_source_mode == "SPECIFIC":
                specified_workspace_name = GlobalProterties.specific_workspace_name()
                if specified_workspace_name:
                    return specified_workspace_name

            if workspace_source_mode == "CUSTOM":
                custom_workspace_folder_path = cls._normalize_workspace_folder_path(
                    GlobalProterties.custom_workspace_folder_path()
                )
                if custom_workspace_folder_path:
                    return os.path.basename(custom_workspace_folder_path.rstrip("\\/"))

            # 独立模式回退：未安装SSMT4时SYNC模式无法解析工作空间名，
            # 若用户已填写自定义目录则直接使用，效果等同于CUSTOM模式
            if not cls.workspacename and not cls.is_ssmt4_available():
                custom_workspace_folder_path = cls._normalize_workspace_folder_path(
                    GlobalProterties.custom_workspace_folder_path()
                )
                if custom_workspace_folder_path:
                    return os.path.basename(custom_workspace_folder_path.rstrip("\\/"))
        except Exception:
            pass

        return cls.workspacename
    
    @classmethod
    def path_workspace_folder(cls):
        try:
            if GlobalProterties.workspace_source_mode() == "CUSTOM":
                custom_workspace_folder_path = cls._normalize_workspace_folder_path(
                    GlobalProterties.custom_workspace_folder_path()
                )
                if custom_workspace_folder_path:
                    return cls._resolve_custom_workspace_data_folder(custom_workspace_folder_path)
                return ""

            # 独立模式回退：未安装SSMT4时ssmtlocation为空，SYNC/SPECIFIC模式拼不出有效路径，
            # 若用户已填写自定义目录则直接使用，避免拼出无意义的相对路径
            if not cls.ssmtlocation and not cls.is_ssmt4_available():
                custom_workspace_folder_path = cls._normalize_workspace_folder_path(
                    GlobalProterties.custom_workspace_folder_path()
                )
                if custom_workspace_folder_path:
                    return cls._resolve_custom_workspace_data_folder(custom_workspace_folder_path)
                return ""
        except Exception:
            pass

        return os.path.join(GlobalConfig.path_current_game_total_workspace_folder(), cls.get_workspace_name() + "\\")
    
    @classmethod
    def path_generate_mod_folder(cls):
        # 如果用户勾选了使用指定文件夹，那就返回指定文件夹位置，否则返回我们的默认位置。
        # 但是这里有个问题就是SkipIB和VSCheck不会生成在指定位置。
        if GlobalProterties.use_specific_generate_mod_folder_path():
            return GlobalProterties.generate_mod_folder_path()
        elif cls.current_game_migoto_folder:
            # 确保用的时候直接拿到的就是已经存在的目录
            ssmt_generated_mod_folder_path = os.path.join(GlobalConfig.path_mods_folder(),"SSMTGeneratedMod\\")
            generate_mod_folder_path = os.path.join(ssmt_generated_mod_folder_path, cls.get_workspace_name() + "\\")
            if not os.path.exists(generate_mod_folder_path):
                os.makedirs(generate_mod_folder_path)
            return generate_mod_folder_path
        else:
            # 独立模式回退：没有installDir时，生成到当前工作空间目录下的GeneratedMod文件夹
            # 此方法只在导出流程中被调用，因此可以在这里按需创建目录
            workspace_folder_path = cls.path_workspace_folder()
            if not workspace_folder_path:
                return ""
            generate_mod_folder_path = os.path.join(workspace_folder_path, "GeneratedMod\\")
            if not os.path.exists(generate_mod_folder_path):
                os.makedirs(generate_mod_folder_path)
            return generate_mod_folder_path
    
    @classmethod
    def path_extract_gametype_folder(cls,draw_ib:str,gametype_name:str):
        return os.path.join(GlobalConfig.path_workspace_folder(), draw_ib + "\\TYPE_" + gametype_name + "\\")
    
    @classmethod
    def path_generatemod_buffer_folder(cls):
        from ..blueprint.export_helper import BlueprintExportHelper
        buffer_folder_name = BlueprintExportHelper.get_current_buffer_folder_name()
        buffer_path = os.path.join(GlobalConfig.path_generate_mod_folder(), buffer_folder_name + "\\")
        if not os.path.exists(buffer_path):
            os.makedirs(buffer_path)
        return buffer_path
    
    @classmethod
    def path_generatemod_texture_folder(cls,draw_ib:str):

        texture_path = os.path.join(GlobalConfig.path_generate_mod_folder(),"Textures\\")
        if not os.path.exists(texture_path):
            os.makedirs(texture_path)
            print("GlobalConfig: 已创建贴图输出目录: " + texture_path + " (DrawIB: " + str(draw_ib) + ")")
        else:
            print("GlobalConfig: 使用已有贴图输出目录: " + texture_path + " (DrawIB: " + str(draw_ib) + ")")
        return texture_path
    
    @classmethod
    def path_appdata_local(cls):
        return os.path.join(os.environ['LOCALAPPDATA'])
    
    @classmethod
    def path_ssmt4_global_configs_folder(cls):
        return os.path.join(GlobalConfig.path_appdata_local(),"SSMT4GlobalConfigs\\")

    # 定义基础的Json文件路径
    @classmethod
    def path_main_json_ssmt4(cls):
        return os.path.join(GlobalConfig.path_ssmt4_global_configs_folder(), "settings.json")
    



    
