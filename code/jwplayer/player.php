<?php
use PhpParser\Node\Stmt\If_;

/*
 *########     (C) 2018 DeepKeeper     ########
 *########     Time:        2018-1-3                    ########
 *########      File:        UTF-8 player.php                 ########
 *########     Author:     Email: DeepKeeper@qq.com     ########
 */
error_reporting ( 0 );
set_magic_quotes_runtime ( 0 );
define ( 'S_ROOT', dirname ( __FILE__ ) . DIRECTORY_SEPARATOR );
define ( 'BG_ROOT', S_ROOT  . 'backgroud' . DIRECTORY_SEPARATOR );

include_once(S_ROOT.'./function.php');

$widht  = empty($_GET['w'])? 960 : intval($_GET['w']);
$height = empty($_GET['h'])? 540 : intval($_GET['h']);
$start = empty($_GET['as'])? false : _bool($_GET['as']);
$start = $start ? 'true' : 'false';
$url   = empty($_GET['u'])? 'rtmp://live.hailigu.com/app/tt' : trim($_GET['u']);
$bg    = empty($_GET['bg'])? '' : trim($_GET['bg']);
$siteUrl = getSiteUrl();
echo is_readable(BG_ROOT.$bg);
if (file_exists(BG_ROOT.$bg)){
    $bg = $siteUrl .'background/'.$bg;
}else{
    $bg = $siteUrl.'background/bg.jpg';
}

header("Content-type: text/html; charset=utf-8");
$html = print<<<END
<script type="text/javascript" src="{$siteUrl}jwplayer/jwplayer.js"></script>

<div id="container">Loading the player ...</div>
    <script type="text/javascript">
        jwplayer("container").setup({
        sources: [
            {
                file: "{$url}"
            }
        ],
        image: "{$bg}",
        autostart: {$start},
        width: {$widht},
        height: {$height},
        primary: "flash"
});
</script>
END;
echo $html;       

