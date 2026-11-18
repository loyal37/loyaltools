"""Run in Blender: --background --factory-startup --python this_file.py."""
from pathlib import Path
import importlib,sys,types,unittest
try:
    import bpy
except ImportError:
    raise unittest.SkipTest('requires Blender 4.5')

ROOT=Path(__file__).resolve().parents[1]
PACKAGE='loyaltools_preprocess_test'
module=types.ModuleType(PACKAGE);module.__path__=[str(ROOT)];sys.modules[PACKAGE]=module
physics=importlib.import_module(PACKAGE+'.efmi_physics')
preprocess=importlib.import_module(PACKAGE+'.blueprint.preprocess')

def main():
    physics.register()
    scene=bpy.context.scene
    settings=scene.loyaltools_efmi_physics
    helper=preprocess.PreProcessHelper
    preprocess.GlobalConfig.logic_name=preprocess.LogicName.EFMI
    preprocess.GlobalProterties.enable_non_mirror_workflow=classmethod(lambda cls:False)
    preprocess.GlobalProterties.apply_all_modifiers=classmethod(lambda cls:True)
    mesh=bpy.data.meshes.new('Physics source mesh')
    mesh.from_pydata([(0,0,0),(0,1,0),(1,1,0)],[],[(0,1,2)])
    source=bpy.data.objects.new('Physics source',mesh);scene.collection.objects.link(source)
    source['LoyalTools:EFMIMergedSkeleton']=True
    owned=bpy.data.objects.new('Physics authoring rig',bpy.data.armatures.new('Physics bones'))
    owned['LoyalTools:EFMIPhysicsRig']=True;scene.collection.objects.link(owned)
    normal=bpy.data.objects.new('Ordinary armature',bpy.data.armatures.new('Ordinary bones'))
    scene.collection.objects.link(normal)
    preview=source.modifiers.new('Physics preview','ARMATURE');preview.object=owned
    ordinary=source.modifiers.new('User armature','ARMATURE');ordinary.object=normal
    entry=settings.objects.add();entry.uid='preview-test';entry.object=source;entry.rig=owned
    source['LoyalTools:EFMIPhysicsObject']=entry.uid
    source['LoyalTools:EFMIPhysicsOwnedGroups']='[]'
    owned['LoyalTools:EFMIPhysicsOwner']=entry.uid
    # The optional module is disabled by default; no legacy modifier is changed.
    assert not settings.enabled
    assert helper._strip_efmi_physics_preview_modifiers(source)==0
    assert len(source.modifiers)==2
    settings.enabled=True
    # Preprocessing operates on a disposable mesh copy, never the user's source.
    copy=source.copy();copy.data=source.data.copy();scene.collection.objects.link(copy)
    preprocess.PreProcessCache.tag_runtime_copy(copy,source.name,source.name,'fixture-hash')
    helper.register_copy_result(source.name,copy.name)
    assert helper._strip_efmi_physics_preview_modifiers(copy)==1
    assert [m.name for m in copy.modifiers]==['User armature']
    assert [m.name for m in source.modifiers]==['Physics preview','User armature']
    # A marked rig does not grant permission to change an unregistered mesh.
    unregistered=source.copy();unregistered.data=source.data.copy();scene.collection.objects.link(unregistered)
    unregistered.pop('LoyalTools:EFMIPhysicsObject',None)
    assert helper._strip_efmi_physics_preview_modifiers(unregistered)==0
    assert len(unregistered.modifiers)==2
    # Preview modifier baking and stripping must not share a preprocess cache.
    cache=preprocess.PreProcessCache
    enabled_hash=cache.compute_object_hash(source.name)
    settings.enabled=False
    disabled_hash=cache.compute_object_hash(source.name)
    assert enabled_hash!=disabled_hash
    unrelated_off_hash=cache.compute_object_hash(unregistered.name)
    settings.enabled=True
    assert cache.compute_object_hash(source.name)==enabled_hash
    assert cache.compute_object_hash(unregistered.name)==unrelated_off_hash
    # This is an EFMI feature; a scene carrying stale settings in another game
    # keeps the original preprocessing contract.
    preprocess.GlobalConfig.logic_name=preprocess.LogicName.ZZMI
    assert helper._strip_efmi_physics_preview_modifiers(source)==0
    assert len(source.modifiers)==2
    physics.unregister()
    print('EFMI_PHYSICS_PREPROCESS_TESTS_OK')

if __name__=='__main__':main()
